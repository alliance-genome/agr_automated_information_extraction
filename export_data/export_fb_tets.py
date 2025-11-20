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
    """Writes text to the top of a file, preserving existing content."""
    try:
        with open(file_path, 'r') as file:
            existing_content = file.read()
    except FileNotFoundError:
        existing_content = ""

    with open(file_path, 'w') as file:
        file.write(text_to_add + existing_content)


def get_data(table_name: str):
    # Get the current date
    today = datetime.now()

    # Subtract 6 days to get a week
    seven_days_ago = today - timedelta(days=6)

    # Format the date for SQL (YYYY-MM-DD)
    sql_date = seven_days_ago.strftime('%Y-%m-%d')
    db_session = create_postgres_session(False)
    if table_name == 'tet':

        query = f"""SELECT tet.date_created, tet.topic, tet.confidence_level, cr.curie
                      FROM topic_entity_tag tet, cross_reference cr, topic_entity_tag_source s
                        WHERE tet.date_created >= '{sql_date}'
                             AND s.topic_entity_tag_source_id = tet.topic_entity_tag_source_id
                             AND cr.reference_id = tet.reference_id
                             AND cr.curie_prefix = 'PMID'
                             AND s.source_method = 'abc_document_classifier'
                             AND tet.species in ('NCBITaxon:7227', 'NCBITaxon:7214')
                             order by tet.date_created desc;"""
        print(query)
    else:
        query = f"""SELECT mit.date_created, mit.curation_tag, mit.confidence_score, cr.curie
                      FROM manual_indexing_tag mit, cross_reference cr
                        WHERE mit.date_created >= '{sql_date}'
                             AND mit.reference_id = cr.reference_id
                             AND cr.curie_prefix = 'PMID'
                             AND mit.mod_id = 1
                             AND mit.validation_by_biocurator is NULL
                             order by mit.date_created desc"""
        print(query)

    rs = db_session.execute(text(query))
    rows = rs.fetchall()
    return rows


def dump_tet():
    conversions = {"NEG": 'neg',
                   "High": 'high',
                   "Low": 'low',
                   "Med": 'medium',
                   "MEDIUM": 'medium',
                   "LOW": 'low',
                   "HIGH": 'high',
                   }
    atp_to_flag = {"ATP:0000152": 'disease',
                   "ATP:0000013": 'new_transg',
                   "ATP:0000006": 'new_al',
                   "ATP:0000069": 'phys_int',
                   "ATP:0000207": 'nocur',
                   "ATP:0000005": 'gene'}
    atp_to_dept = {"ATP:0000152": 'dis',
                   "ATP:0000013": 'cam',
                   "ATP:0000006": 'cam',
                   "ATP:0000069": 'harv',
                   "ATP:0000207": 'cam',
                   "ATP:0000005": 'cam'}

    pos = ""
    neg = ""
    for table_name in ('tet', 'mi'):  # topic_entity_tag, Manual_indexing
        rows = get_data(table_name)
        for row in rows:
            conf_level = 'LOW'
            # do not crash on keyword error. Give message and continue.
            if row[1] not in atp_to_flag:
                atp_to_flag[row[1]] = f'UNKNOWN_FLAG_{row[1]}'
            if row[2] not in conversions:
                # Using guessed score levels until told otherwise
                # taken from method get_confidence_level in classifier
                if type(row[2]) == float:
                    ## add loop to set confidence level to NEG for manual_indexing_tag results where mit.confidence_score = 0 so that they can be put in the negative pot
                    if row[2] == 0:
                        conf_level = "NEG"
                    elif row[2] < 0.667:
                        conf_level = "LOW"
                    elif row[2] < 0.833:
                        conf_level = "MEDIUM"
                    else:
                        conf_level = "HIGH"
                else:
                    conf_level = f'UNKNOWN_LEVEL_{row[2]}'
            else:
                conf_level = conversions[row[2]]
            if row[1] not in atp_to_dept:
                atp_to_dept[row[1]] = f'UNKNOWN_ATP_{row[1]}'
            ## use conf_level rather than row[2] to partition data into positive/negative pots so that negative manual_indexing_tag results go in the negative pot
            if conf_level != 'NEG':
                pos += (
                    f"{row[0].strftime('%y%m%d')}\t\t"
                    f"{row[3][5:]}\t"
                    f"{atp_to_flag[row[1]]}:{conf_level}\t"
                    f"{atp_to_dept[row[1]]}\n"
                )
            else:
                neg += (
                    f"{row[0].strftime('%y%m%d')}\t\t"
                    f"{row[3][5:]}\t"
                    f"{atp_to_flag[row[1]]}:{conf_level}\t"
                    f"{atp_to_dept[row[1]]}\n"
                )

    if pos:
        print("Updating positive data")
        print(pos)
        write_to_top('/curation_status/textmining_positive_ABC.txt', pos)
    else:
        print("No positive data found")
    if neg:
        print("Updating negative data")
        write_to_top('/curation_status/textmining_negative_ABC.txt', neg)
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
