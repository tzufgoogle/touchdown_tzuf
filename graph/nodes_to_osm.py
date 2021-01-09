import pandas as pd
import osmnx as ox
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import config

TOURISM = [ 'gallery',
            'guest_house',
            'hostel',
            'hotel',
            'motel',
            'museum']
LEISURE = ['park', 'playground', 'stadium']
BUILDING = ['hotel',
            'supermarket',
            'warehouse',
            'cathedral',
            'chapel',
            'church',
            'mosque',
            'synagogue',
            'hospital',
            'kindergarten',
            'school',
            'university',
            'train_station',
            'stadium',
            'bridge']

TAGS = {'name': True,
        'amenity': True,
        'wikidata': True,
        'wikipedia': True,
        'shop': True,
        'brand': True,
        'tourism': TOURISM,
        'leisure': LEISURE,
        'building': BUILDING
        }

POI_TAGS = {'name': True,
            'amenity': True,
            'shop': True,
            'brand': True,
            'tourism': TOURISM,
            'leisure': LEISURE,
            }
DIST = 50
Counter = 0

print ("Start")
poi = ox.pois.pois_from_place("Manhattan", TAGS, which_result=None)
poi = poi[list(POI_TAGS.keys())+["geometry"]]

class OSMImageNode:
    def __init__(self, node_id, poi, lon, lat, heading):
        self.node_id = node_id
        self.poi = poi
        self.lon = lon
        self.lat = lat
        self.heading = heading


def get_distance_between_geometries(geometry, point):
  '''Calculate the distance between point and polygon in meters.
  Arguments:
    route: The line that length calculation will be performed on.
    point: point to measure distance from polygon.
  Returns:
    The distance between point and polygon in meters.
  '''
  if isinstance(geometry, Point):
    return get_distance_between_points(geometry, point)
  else:
    return get_distance_between_point_to_geometry(geometry, point)

def get_distance_between_point_to_geometry(
  geometry, point):
  '''Calculate the distance between point and polygon in meters.
  Arguments:
    route: The line that length calculation will be performed on.
    point: point to measure distance from polygon.
  Returns:
    The distance between point and polygon in meters.
  '''
  dist_min = float("Inf")
  if isinstance(geometry, MultiPolygon):
    coords = [coord for poly in geometry for coord in poly.exterior.coords]
  elif isinstance(geometry, Polygon):
    coords = geometry.exterior.coords
  else:
    coords = geometry.coords
  for coord in coords:
    point_current = Point(coord)
    dist = get_distance_between_points(point, point_current)
    if dist_min > dist:
      dist_min = dist
  return dist_min


def get_distance_between_points(point_1, point_2):
    '''Calculate the line length in meters.
    Arguments:
      point_1: The point to calculate the distance from.
      point_2: The point to calculate the distance to.
    Returns:
      Distance length in meters.
    '''

    dist = ox.distance.great_circle_vec(
        point_1.y, point_1.x, point_2.y, point_2.x)

    return dist


def add_poi(row):
    lat, lon =row.lat, row.lon
    global Counter
    Counter+=1
    print (Counter)
    point = Point(lon, lat)
    poi_within_distance = poi[poi.geometry.apply(lambda x: get_distance_between_geometries(x,point)< DIST)]

    tags_dict = {}
    for tag in POI_TAGS:
        if tag in poi_within_distance:
            if tag not in tags_dict:
                tags_dict[tag] = []

            tags_dict[tag] = list(set(poi_within_distance[poi_within_distance[tag].notnull()][tag].tolist()))

    row['poi'] = tags_dict

    return row

if __name__ == '__main__':
    os.chdir('../')
    node_file = config.paths['node']
    ds = pd.read_csv(node_file,
                     names=["panoid", "heading", "lat", "lon"])

    print ("!", ds.shape)
    ds_with_poi = ds.apply(add_poi, axis=1)
    ds_with_poi.to_json(config.paths['node_poi'])
    print ("!!!")
    print ("END")

