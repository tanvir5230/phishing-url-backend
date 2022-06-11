from audioop import add
from unittest import result
from urllib.parse import urlparse
def is_ip(address):
    network_Location = urlparse(address).netloc
    result = not(network_Location.split('.')[-1].isalpha());
    result = -1 if result else 1;#1 means the url doesn't contain IP address and -1 means the opposite
    return result;

def url_length_feature(address):
    length_of_url = len(address);
    if(length_of_url < 54):
        result = 1;#1 means length is less than 54, legitimate
    elif(length_of_url>= 54 and length_of_url <=75):
        result = 0;#length is between 54 to 75 and suspicious
    else:
        result = -1;
    return result;#length is greater that 75 and phishing
def having_At_Symbol(address):
    result = "@" in address;
    result = -1 if result else 1; #-1 means address contains @ and 1 means opposite
    return result
def prefix_Suffix(address):
    result =  "-" in address;
    result = -1 if result else 1; #-1 means address contains - and 1 means opposite
    return result
