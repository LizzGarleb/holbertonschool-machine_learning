#!/usr/bin/env python3
"""
  Module insert_school
"""


def insert_school(mongo_collection, **kwargs):
    """
      Insert a new document in a collection based on kwargs.
    """
    document = mongo_collection.insert_one(kwargs)
    return (document.inserted_id)
