#!/usr/bin/env python3
"""
  Script that displays the upcoming launch
"""
import requests


if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    results = requests.get(url).json()
    dateCheck = float('inf')
    launchName = None
    rocket = None
    launchPad = None
    location = None
    for launch in results:
        launchDate = launch.get('date_unix')
        if launchDate < dateCheck:
            dateCheck = launchDate
            date = launch.get('date_local')
            launchName = launch.get('name')
            rocket = launch.get('rocket')
            launchPad = launch.get('launchpad')
    if rocket:
        rocket = requests.get('https://api.spacexdata.com/v4/rockets/{}'.
                              format(rocket)).json().get('name')
    if launchPad:
        launchpad = requests.get('https://api.spacexdata.com/v4/launchpads/{}'.
                                 format(launchPad)).json()
        launchPad = launchpad.get('name')
        location = launchpad.get('locality')
    print('{} ({}) {} - {} ({})'.format(launchName,
          date, rocket, launchPad, location))
