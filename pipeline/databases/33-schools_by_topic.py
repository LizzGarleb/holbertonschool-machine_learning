#!/usr/bin/env python3
"""
    Module schools_by_topic
"""


def schools_by_topic(mongo_collection, topic):
    """
        List of school having a specific topic.
    """
    schools = []
    documents = mongo_collection.find({'topics': {'$all': [topic]}})
    for doc in documents:
        schools.append(doc)
    return schools
