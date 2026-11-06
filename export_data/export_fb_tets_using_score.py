from sqlalchemy import text
from os import environ
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import traceback
from agr_literature_service.lit_processing.utils.report_utils import send_report


def create_postgres_engine(verbose):

    """Connect to database."""
    USER = environ.get('BLUE_USERNAME', 'postgres')
    PASSWORD = environ.get('BLUE_PASSWORD')
    SERVER = environ.get('BLUE_HOST', 'literature-prod.cmnnhlso7wdi.us-east-1.rds.amazonaws.com')
    PORT = environ.get('BLUE_PORT', '5432')
    DB = environ.get('BLUE_DATABASE', 'literature')

    # Create our SQL Alchemy engine from our environmental variables.
    engine_var = 'postgresql://' + USER + ":" + PASSWORD + '@' + SERVER + ':' + PORT + '/' + DB
    # future=True is recommended for 2.0-style behavior.  But referencefile unit test would fail, so removed it.
    engine = create_engine(engine_var)

    return engine


def create_postgres_session(verbose):

    engine = create_postgres_engine(verbose)

    # SQLAlchemy 2.0 recommends using 'autocommit=False' explicitly in sessionmaker
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = Session()

    return session


def write_to_top(file_path, text_to_add):
    """Writes text to the top of a file, preserving existing content.

    The file is FlyBase's SVN working copy, so it is only written when FB_SVN_COMMIT is "true",
    which only the production GoCD pipeline sets. Any other run (stage, a test) prints what it
    would have added to stdout instead.
    """
    if environ.get('FB_SVN_COMMIT') != 'true':
        print(f'FB_SVN_COMMIT is not "true", so {file_path} is not written. It would add:')
        print(text_to_add, end='')
        return
    print(f"Adding to {file_path}:")
    print(text_to_add, end='')
    try:
        with open(file_path, 'r') as file:
            existing_content = file.read()
    except FileNotFoundError:
        existing_content = ""

    with open(file_path, 'w') as file:
        file.write(text_to_add + existing_content)


def get_data():
    # Get the current date
    today = datetime.now()

    # Subtract 6 days to get a week
    seven_days_ago = today - timedelta(days=6)

    # Format the date for SQL (YYYY-MM-DD)
    sql_date = seven_days_ago.strftime('%Y-%m-%d')
    db_session = create_postgres_session(False)
    # changed query to get tet.confidence_score instead of tet.confidence_level
    query = f"""SELECT tet.date_created, tet.topic, tet.confidence_score, cr.curie, tet.validation_by_professional_biocurator
                  FROM topic_entity_tag tet, cross_reference cr, tag_source s
                    WHERE tet.date_created >= '{sql_date}'
                         AND s.id = tet.tag_source_id
                         AND cr.reference_id = tet.reference_id
                         AND cr.curie_prefix = 'PMID'
                         AND s.source_method = 'abc_document_classifier'
                         AND tet.species in ('NCBITaxon:7227', 'NCBITaxon:7214')
                         AND tet.validation_by_professional_biocurator IS DISTINCT FROM 'validated_wrong'
                         order by tet.date_created desc;"""
    print(query)

    rs = db_session.execute(text(query))
    rows = rs.fetchall()
    return rows


def dump_tet():
    atp_to_flag = {"ATP:0000152": 'disease',
                   "ATP:0000013": 'new_transg',
                   "ATP:0000006": 'new_al',
                   "ATP:0000069": 'phys_int',
                   "ATP:0000402": 'nocur',  # SCRUM-6669/6670: "genetic data" tags from the no genetic data model
                   }
    atp_to_dept = {"ATP:0000152": 'dis',
                   "ATP:0000013": 'cam',
                   "ATP:0000006": 'cam',
                   "ATP:0000069": 'harv',
                   "ATP:0000402": 'cam',
                   }

    pos = ""
    neg = ""
    for row in get_data():
        # do not crash on keyword error. Give message and continue.
        if row[1] not in atp_to_flag:
            atp_to_flag[row[1]] = f'UNKNOWN_FLAG_{row[1]}'
        if row[1] not in atp_to_dept:
            atp_to_dept[row[1]] = f'UNKNOWN_ATP_{row[1]}'
        ## put anything with a score of 0 in the negative pot, everything else goes in the positive pot
        ## in both cases, print the confidence_score after "flag:"
        if row[2] == 0:
            neg += (
                f"{row[0].strftime('%y%m%d')}\t\t"
                f"{row[3][5:]}\t"
                f"{atp_to_flag[row[1]]}:{row[2]}\t"
                f"{atp_to_dept[row[1]]}\t"
                f"{row[4]}\n"
            )
        else:
            pos += (
                f"{row[0].strftime('%y%m%d')}\t\t"
                f"{row[3][5:]}\t"
                f"{atp_to_flag[row[1]]}:{row[2]}\t"
                f"{atp_to_dept[row[1]]}\t"
                f"{row[4]}\n"
            )

    if pos:
        print("Updating positive data")
        write_to_top('/curation_status/textmining_positive_ABC_using_score.txt', pos)
    else:
        print("No positive data found")
    if neg:
        print("Updating negative data")
        write_to_top('/curation_status/textmining_negative_ABC_using_score.txt', neg)
    else:
        print("No negative data found")


if __name__ == "__main__":
    try:
        dump_tet()
    except Exception as e:
        formatted_traceback = traceback.format_tb(e.__traceback__)
        subject = "Failed generating FB flags report"
        message = "<h>Failed generating FB flags report:</h><br><br>\n\n"
        message += f"Exception: {str(e)}<br>\nStacktrace:<br>{''.join(formatted_traceback)}"

        send_report(subject, message)
        exit(-1)
