import re
from urllib.parse import parse_qs, urlparse


# Fetch the query string from the URL
def split_query_string(url):
    # Processing by urlparse
    parsed_url = urlparse(url)
    # Get the query string from parsed URL
    query_string = parsed_url.query
    # Generate a set of the query string
    query_params = parse_qs(query_string)

    return query_params


# Remove the query string from the URL
def remove_query_string(url: str) -> str:
    # Use regular expression to remove query string
    result = re.sub(r'\?.*', '', url)

    return result


# Extract the base URL from the URL
def extract_base_url(url):
    # Define the pattern and init variable
    pattern = r"((?<=\/\/)[^\/]+)"
    base_url = ''

    # Extract the base URL with regex
    match = re.search(pattern, url)
    if match:
        base_url = match.group(1)

    return base_url