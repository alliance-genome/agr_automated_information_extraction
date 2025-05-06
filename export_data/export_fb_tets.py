from sqlalchemy import text
from os import environ
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta


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


def dump_tet():
    conversions = {"NEG": 'neg',
                   "High": 'high',
                   "Low": 'low',
                   "Med": 'medium'}
    atp_to_flag = {"ATP:0000152": 'disease',
                   "ATP:0000013": 'new_transg',
                   "ATP:0000006": 'new_al',
                   "ATP:0000069": 'phys_int',
                   "ATP:0000207": 'nocur'}
    atp_to_dept = {"ATP:0000152": 'dis',
                   "ATP:0000013": 'cam',
                   "ATP:0000006": 'cam',
                   "ATP:0000069": 'harv',
                   "ATP:0000207": 'cam'}
    db_session = create_postgres_session(False)

    # Get the current date
    today = datetime.now()

    # Subtract 7 days
    seven_days_ago = today - timedelta(days=7)

    # Format the date for SQL (YYYY-MM-DD)
    sql_date = seven_days_ago.strftime('%Y-%m-%d')
    query = f"""SELECT tet.date_created, tet.topic, tet.confidence_level, cr.curie
                  FROM topic_entity_tag tet, cross_reference cr
                    WHERE tet.date_created >= '{sql_date}'
                         AND cr.reference_id = tet.reference_id
                         AND cr.curie_prefix = 'PMID'
                         AND tet.species in ('NCBITaxon:7227', 'NCBITaxon:7214')
                         order by tet.date_created desc"""
    print(query)
    rs = db_session.execute(text(query))
    rows = rs.fetchall()
    pos = ""
    neg = ""
    for row in rows:
        if row[2] != 'NEG':
            pos += f"{row[0].strftime('%y%m%d')}\t\t{row[3][5:]}\t{atp_to_flag[row[1]]}:{conversions[row[2]]}\t{atp_to_dept[row[1]]}\n"
        else:
            neg += f"{row[0].strftime('%y%m%d')}\t\t{row[3][5:]}\t{atp_to_flag[row[1]]}:{conversions[row[2]]}\t{atp_to_dept[row[1]]}\n"

    if pos:
        print("Updating positive data")
        print(pos)
        write_to_top('./textmining_positive_ABC.txt', pos)
    else:
        print("No positive data found")
    if neg:
        print("Updating negative data")
        write_to_top('./textmining_negative_ABC.txt', neg)
    else:
        print("No negative data found")


if __name__ == "__main__":
    dump_tet()
