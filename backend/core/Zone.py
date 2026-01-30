from shapely.geometry import MultiPoint
import numpy as np
from shapely.geometry.base import BaseGeometry
from sklearn.cluster import DBSCAN

from .constants import ZoneType


class Zone:
    """This is class representation of a zone.
    """

    def __init__(self, zone_id, zone_type: ZoneType):
        self.points = []
        self.centroid = []
        self.id = zone_id
        self.zoneType = zone_type
        self.area = []
        self.cycle_ids = []

        self.speedLimit = None
        self.tractionLevel = None

    def updatePoints(self, points, cycle_ids=None):
        """this function updates the zone points

        :param points: list of x, y coordinates of points
        :type points: list
        :param cycle_ids: list of cycle IDs corresponding to points
        :type cycle_ids: list
        """
        self.points.extend(points)
        # Remove duplicates while preserving order
        self.points = list(dict.fromkeys(self.points))
        if cycle_ids:
            self.cycle_ids = list(set(self.cycle_ids + cycle_ids))


    def updateProperties(self, check_clusters=False, eps=50):
        """this function computes the centroid and polygon surrounding the zone area
        """        
        if self.points:
            points_list = []
            if check_clusters:
                dbscan = DBSCAN(eps=eps)
                dbscan.fit(self.points)
                labels = dbscan.labels_
                unique_values = np.unique(labels)
                if len(unique_values) > 1:
                    for value in unique_values:
                        points_list.append([self.points[i] for i in np.where(labels == value)[0]])
                else:
                    points_list.append(self.points)
            else:
                points_list.append(self.points)

            for points in points_list:
                coords = list(map(list, zip(*points)))
                x, y = coords[0], coords[1]
                if len(coords) > 2:
                    z = coords[2]
                    self.centroid.append((np.mean(x), np.mean(y), np.mean(z)))
                else:
                    self.centroid.append((np.mean(x), np.mean(y)))
                # Use only (x, y) for 2D geometry operations
                xy_points = list(zip(x, y))
                if len(set(xy_points)) > 2:
                    self.area.append(MultiPoint(xy_points).convex_hull)
                elif len(set(xy_points)) >= 1:
                    self.area.append(MultiPoint(xy_points).buffer(5))

    def isZoneInArea(self, x_min, x_max, y_min, y_max):
        """This function checks whether zone is present in particular area.

        :param x_min: minimum x value of rectangular area
        :type x_min: float
        :param x_max: maximum x value of rectangular area
        :type x_max: float
        :param y_min: minimum y value of rectangular area
        :type y_min: float
        :param y_max: maximum y value of rectangular area
        :type y_max: float
        :return: True if zone is inside the area else False
        :rtype: boolean
        """
        pass

    def isZoneInPolygon(self, polygon):
        """checks whether zone area intersects with polygon area

        :param polygon: polygon which represents the location
        :type polygon: Shapely.Polygon
        :return: True if zone belongs to that location
        :rtype: bool
        """
        is_in = False
        for area in self.area:
            if area.intersects(polygon):
                return True
        return False

    def isSameZone(self, new_zone):
        for new_zone_area in new_zone.area:
            for area in self.area:
                if area.intersects(new_zone_area):
                    return True
        return False

    def to_dict(self):
        area = (
            Zone.export_polygon_data(self.area)
            if not isinstance(self.area, list)
            else [Zone.export_polygon_data(area) for area in self.area]
        )

        area_exterior = [a["exterior"] for a in area if a] if area else None
        return {
            "id": self.id,
            "zoneType": self.zoneType.value,
            "centroid": self.centroid,
            "points": self.points,
            "area_exterior": area_exterior,
            "cycle_ids": list(set(self.cycle_ids)) if self.cycle_ids else None,
        }

    def export_polygon_data(polygon: BaseGeometry):
        if polygon.geom_type != "Polygon":
            return None

        return {
            "type": polygon.geom_type,  # "Polygon"
            "exterior": list(polygon.exterior.coords),
            "interiors": [list(interior.coords) for interior in polygon.interiors],
            "area": polygon.area,
            "length": polygon.length,
            "centroid": {"x": polygon.centroid.x, "y": polygon.centroid.y},
            "bounds": polygon.bounds,
            "envelope": list(polygon.envelope.exterior.coords),
            "wkt": polygon.wkt,  # Optional: Well-known text format
            "geo_interface": polygon.__geo_interface__,  # Optional: Full GeoJSON-like structure
        }