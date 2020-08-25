#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from collections import defaultdict

# hwrt modules
from hwrt import handwritten_data
from hwrt import utils
from hwrt import data_analyzation_metrics
from hwrt import geometry


class TrainingCount(object):
    """Analyze how many training examples exist for each recording."""

    def __init__(self, filename="creator.csv"):
        self.filename = data_analyzation_metrics.prepare_file(filename)

    def __repr__(self):
        return "TrainingCount(%s)" % self.filename

    def __str__(self):
        return "TrainingCount(%s)" % self.filename

    def __call__(self, raw_datasets):
        write_file = open(self.filename, "a")
        write_file.write("symbol,trainingcount\n")  # heading

        print_data = defaultdict(int)
        start_time = time.time()
        for i, raw_dataset in enumerate(raw_datasets):
            if i % 100 == 0 and i > 0:
                utils.print_status(len(raw_datasets), i, start_time)
            print_data[raw_dataset["handwriting"].formula_in_latex] += 1
        print("\r100%" + "\033[K\n")
        # Sort the data by highest value, descending
        print_data = sorted(print_data.items(), key=lambda n: n[1], reverse=True)
        # Write data to file
        write_file.write("total,%i\n" % sum([value for _, value in print_data]))
        for userid, value in print_data:
            write_file.write("%s,%i\n" % (userid, value))
        write_file.close()


def get_bounding_box_distance(raw_datasets):
    """Get the distances between bounding boxes of strokes of a single symbol.
       Can only be applied to recordings with at least two strokes.
    """

    # TODO: Deal with http://www.martin-thoma.de/write-math/symbol/?id=167
    # 193
    # 524

    def _get_stroke_bounding_box(stroke):
        """Get the bounding box of a stroke. A stroke is a list of points
           {'x': 123, 'y': 456, 'time': 42} and a bounding box is the smallest
           rectangle that contains all points.
        """
        min_x, max_x = stroke[0]["x"], stroke[0]["x"]
        min_y, max_y = stroke[0]["y"], stroke[0]["y"]
        #  if len(stroke) == 1: ?
        for point in stroke:
            min_x = min(point["x"], min_x)
            max_x = max(point["x"], max_x)
            min_y = min(point["y"], min_y)
            max_y = max(point["y"], max_y)
        minp = geometry.Point(min_x, min_y)
        maxp = geometry.Point(max_x, max_y)
        return geometry.BoundingBox(minp, maxp)

    def _get_bb_distance(a, b):
        """"Take two bounding boxes a and b and get the smallest distance
            between them.
        """
        points_a = [
            geometry.Point(a.p1.x, a.p1.y),
            geometry.Point(a.p1.x, a.p2.y),
            geometry.Point(a.p2.x, a.p1.y),
            geometry.Point(a.p2.x, a.p2.y),
        ]
        points_b = [
            geometry.Point(b.p1.x, b.p1.y),
            geometry.Point(b.p1.x, b.p2.y),
            geometry.Point(b.p2.x, b.p1.y),
            geometry.Point(b.p2.x, b.p2.y),
        ]
        min_distance = points_a[0].dist_to(points_b[0])
        for pa in points_a:
            for pb in points_b:
                min_distance = min(min_distance, pa.dist_to(pb))
        lines_a = [
            geometry.LineSegment(points_a[0], points_a[1]),
            geometry.LineSegment(points_a[1], points_a[2]),
            geometry.LineSegment(points_a[2], points_a[3]),
            geometry.LineSegment(points_a[3], points_a[0]),
        ]
        lines_b = [
            geometry.LineSegment(points_b[0], points_b[1]),
            geometry.LineSegment(points_b[1], points_b[2]),
            geometry.LineSegment(points_b[2], points_b[3]),
            geometry.LineSegment(points_b[3], points_b[0]),
        ]
        for line_in_a in lines_a:
            for line_in_b in lines_b:
                min_distance = min(min_distance, line_in_a.dist_to(line_in_b))
        return min_distance

    bbfile = open("bounding_boxdist.html", "a")
    start_time = time.time()
    for i, raw_dataset in enumerate(raw_datasets):
        if i % 100 == 0 and i > 0:
            utils.print_status(len(raw_datasets), i, start_time)
        pointlist = raw_dataset["handwriting"].get_pointlist()
        if len(pointlist) < 2:
            continue
        bounding_boxes = []
        for stroke in pointlist:
            # TODO: Get bounding boxes of strokes
            bounding_boxes.append(_get_stroke_bounding_box(stroke))

        got_change = True
        while got_change:
            got_change = False
            i = 0
            while i < len(bounding_boxes):
                j = i + 1
                while j < len(bounding_boxes):
                    if geometry.do_bb_intersect(bounding_boxes[i], bounding_boxes[j]):
                        got_change = True
                        new_bounding_boxes = []
                        p1x = min(bounding_boxes[i].p1.x, bounding_boxes[j].p1.x)
                        p1y = min(bounding_boxes[i].p1.y, bounding_boxes[j].p1.y)
                        p2x = max(bounding_boxes[i].p2.x, bounding_boxes[j].p2.x)
                        p2y = max(bounding_boxes[i].p2.y, bounding_boxes[j].p2.y)
                        p1 = geometry.Point(p1x, p1y)
                        p2 = geometry.Point(p2x, p2y)
                        new_bounding_boxes.append(geometry.BoundingBox(p1, p2))
                        for k in range(len(bounding_boxes)):
                            if k != i and k != j:
                                new_bounding_boxes.append(bounding_boxes[k])
                        bounding_boxes = new_bounding_boxes
                    j += 1
                i += 1

        # sort bounding boxes (decreasing) by size
        bounding_boxes = sorted(
            bounding_boxes, key=lambda bbox: bbox.get_area(), reverse=True
        )

        # Bounding boxes have been merged as far as possible
        # check their distance and compare it with the highest dimension
        # (length/height) of the biggest bounding box
        if len(bounding_boxes) != 1:
            bb_dist = []
            for k, bb in enumerate(bounding_boxes):
                dist_tmp = []
                for j, bb2 in enumerate(bounding_boxes):
                    if k == j:
                        continue
                    dist_tmp.append(_get_bb_distance(bb, bb2))
                bb_dist.append(min(dist_tmp))
            bb_dist = max(bb_dist)
            dim = max([bb.get_largest_dimension() for bb in bounding_boxes])
            if bb_dist > 1.5 * dim:
                # bounding_box_h = raw_dataset['handwriting'].get_bounding_box()
                # bbsize = (bounding_box_h['maxx'] - bounding_box_h['minx']) * \
                #          (bounding_box_h['maxy'] - bounding_box_h['miny'])
                if (
                    raw_dataset["handwriting"].formula_id
                    not in [635, 636, 936, 992, 260, 941, 934, 184]
                    and raw_dataset["handwriting"].wild_point_count == 0
                    and raw_dataset["handwriting"].missing_stroke == 0
                ):
                    # logging.debug("bb_dist: %0.2f" % bb_dist)
                    # logging.debug("dim: %0.2f" % dim)
                    # for bb in bounding_boxes:
                    #     print(bb)
                    #     print("width: %0.2f" % bb.get_width())
                    #     print("height: %0.2f" % bb.get_height())
                    #     print("maxdim: %0.2f" % bb.get_largest_dimension())
                    # bb_dist = []
                    # for k, bb in enumerate(bounding_boxes):
                    #     dist_tmp = []
                    #     for j, bb2 in enumerate(bounding_boxes):
                    #         if k == j:
                    #             continue
                    #         dist_tmp.append(_get_bb_distance(bb, bb2))
                    #     print(dist_tmp)
                    #     bb_dist.append(min(dist_tmp))
                    # raw_dataset['handwriting'].show()
                    # exit()
                    url_base = "http://www.martin-thoma.de/write-math/view"
                    bbfile.write(
                        "<a href='%s/?raw_data_id=%i'>a</a>\n"
                        % (url_base, raw_dataset["handwriting"].raw_data_id)
                    )
    print("\r100%" + "\033[K\n")
