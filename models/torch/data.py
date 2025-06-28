import collections


WebsitePair = collections.namedtuple('WebsitePair', [
    'w1_feat',
    'w2_feat',
    'labels'])

WebsiteNamePair = collections.namedtuple('WebsiteNamePair', [
    'w1',
    'w2'])
