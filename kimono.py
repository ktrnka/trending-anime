import datetime
import logging


def inject_version(endpoint, api_version):
    """
    Inject an API version into the Kimono endpoint URL
    """
    if api_version and api_version >= 0:
        return endpoint.replace("/api/", "/api/{}/".format(api_version))
    else:
        return endpoint


def is_stale(data_date, refresh_days=1):
    """
    Test if the data is stale
    :param data_date: The date of the data
    :param refresh_days: The Kimono refresh interval in days
    :return:
    """
    assert isinstance(data_date, datetime.datetime)
    logger = logging.getLogger(__name__)
    now = datetime.datetime.now()
    diff = now - data_date
    logger.info("Data is %s days old", diff.days)

    if diff.days > refresh_days:
        return True
    return False


def parse_timestamp(timestamp):
    """Parse a Kimono timestamp like Fri Jan 02 2015 22:04:11 GMT+0000 (UTC)"""
    timestamp = timestamp.replace(" GMT+0000 (UTC)", "")
    return datetime.datetime.strptime(timestamp, "%a %b %d %Y %H:%M:%S")