from .readerConfig import reasonMap


class messageLocationMap:
    """This class represents a small square areas on map which are created by dividing complete area of map into a mesh of constant size.
    """    
    def __init__(self, reference_point_data, indexes, x_min, x_max, y_min, y_max, machines, segments, cycles,
                 reason_summary, health_actions, health_summary):
        """constructor

        :param reference_point_data: data about the point which represents the centroid of all the points in that area
        :type reference_point_data: np.array
        :param indexes: indexes of all the points in the area
        :type indexes: list
        :param x_min: minimum x of the area. This value gives details about left hand side boundary.
        :type x_min: float
        :param x_max: maximum x of the area. This value gives details about right hand side boundary.
        :type x_max: float
        :param y_min: minimum y of the area. This value gives details about bottom boundary.
        :type y_min: float
        :param y_max: maximum y of the area. This value gives details about top boundary.
        :type y_max: float
        :param machines: array of machines that passed through this area
        :type machines: ndarray
        :param segments: array of segments which passes through the area
        :type segments: ndarray
        :param cycles: array of cycles which passes through the area
        :type cycles: ndarray
        :param reason_summary: dictionary contains the summary of reasons which occured in this area
        :type reasons: dict
        """        
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.messagesIndex = indexes
        self.referencePoint = tuple(reference_point_data[0:2])
        self.efficiency = reference_point_data[2]
        self.actualSpeed = reference_point_data[3]
        self.expectedSpeed = reference_point_data[4]
        self.machines = machines
        self.segments = segments
        self.cycles = cycles
        self.reasons = list(reason_summary.keys())
        self.reasonSummary = reason_summary
        self.healthActions = health_actions
        self.healthSummary = health_summary

    def getMapInfoByReason(self, selected_reasons=None):
        """returns the summary of selected reasons in the cell of map (small square area)

        :param selected_reasons: list of reasons whose summary is required, defaults to None
        :type selected_reasons: list, optional
        :return: returns average efficiency, average actual speed and average expected speed of all messages of selected
        reasons in the cell
        :rtype: tuple
        """        
        # if selected_reasons and list(set(selected_reasons).intersection(self.reasons)):
        if selected_reasons:
            efficiency, actual_speed, expected_speed, count = 0, 0, 0, 0
            for reason in selected_reasons:
                data = self.reasonSummary.get(reason)
                if data:
                    efficiency += data['efficiency'] * data['frequency']
                    actual_speed += data['actualSpeed'] * data['frequency']
                    expected_speed += data['expectedSpeed'] * data['frequency']
                    count += data['frequency']

            efficiency = efficiency/count
            actual_speed = actual_speed/count
            expected_speed = expected_speed/count

            return efficiency, actual_speed, expected_speed
        else:
            return self.efficiency, self.actualSpeed, self.expectedSpeed

    def getMapInfoByHealthAction(self, selected_health_actions=None):
        if selected_health_actions:
            efficiency, actual_speed, expected_speed, count = 0, 0, 0, 0
            for health_action in selected_health_actions:
                data = self.healthSummary.get(health_action)
                if data:
                    efficiency += data['efficiency'] * data['frequency']
                    actual_speed += data['actualSpeed'] * data['frequency']
                    expected_speed += data['expectedSpeed'] * data['frequency']
                    count += data['frequency']

            efficiency = efficiency / count
            actual_speed = actual_speed / count
            expected_speed = expected_speed / count

            return efficiency, actual_speed, expected_speed
        else:
            return self.efficiency, self.actualSpeed, self.expectedSpeed

    def updateUnitType(self, new_unit_type):
        self.actualSpeed = self.actualSpeed * (5/18) if new_unit_type == 'mps' else self.actualSpeed * (18/5)
        self.expectedSpeed = self.expectedSpeed * (5/18) if new_unit_type == 'mps' else self.expectedSpeed * (18/5)
        # updating reason summary
        for reason_data in self.reasonSummary.values():
            reason_data['actualSpeed'] = reason_data['actualSpeed'] * (5/18) if new_unit_type == 'mps' else reason_data['actualSpeed'] * (18/5)
            reason_data['expectedSpeed'] = reason_data['expectedSpeed'] * (5/18) if new_unit_type == 'mps' else reason_data['expectedSpeed'] * (18/5)
        # updating health summary
        for health_data in self.healthSummary.values():
            health_data['actualSpeed'] = health_data['actualSpeed'] * (5/18) if new_unit_type == 'mps' else health_data['actualSpeed'] * (18/5)
            health_data['expectedSpeed'] = health_data['expectedSpeed'] * (5/18) if new_unit_type == 'mps' else health_data['expectedSpeed'] * (18/5)


    def toDict(self):
        """this function returns the summary of the area as a dictionary

        :return: dictionary containing information about the area
        :rtype: dict
        """        
        return {
            'x': self.referencePoint[0],
            'y': self.referencePoint[1],
            'efficiency': self.efficiency,
            'actualSpeed': self.actualSpeed,
            'expectedSpeed': self.expectedSpeed
        }


class reasonSummary:
    """This class contains the summary of losses occured due to particular reason in a time range/cycle/segment
    """    
    def __init__(self, reason, reason_name, reason_category, parent):
        """constructor

        :param reason: ASLR id
        :type reason: int
        :param reason_name: ASLR name
        :type reason_name: str
        :param reason_category: ASLR Category
        :type reason_category: str
        :param parent: id of parent machine/cycle/segment to which reason summary belongs
        :type parent: int/str
        """        
        self.parentId = parent
        self.actualAslr = reason
        self.actualAslrName = reason_name
        self.actualAslrCategory = reason_category
        self.actualTimeTaken = 0
        self.expectedTimeTaken = 0
        self.loss = None
        self.efficiency = None
        self.painIndex = None
        self.count = 0
        self.children = {}
        self.buckets = []
        self.opportunity = 0

    def addBucketData(self, bucket):
        """adds bucket data to segments reason summary

        :param bucket: bucket which is to be added
        :type bucket: lossBucket
        """        
        self.actualTimeTaken += bucket.actualTimeTaken
        self.expectedTimeTaken += bucket.expectedTimeTaken
        self.opportunity += bucket.bucketLoss if not bucket.isPowerLimited and bucket.bucketLoss >= 0 else 0
        self.children.setdefault(bucket.bucket_id, []).extend(bucket.messageIndexes)
        self.buckets.append(bucket)
        self.count += 1

    def addSegmentData(self, segment_data):
        """adds reason summary of segment data to get cycle's reason summary

        :param segment_data: reason summary data of segment which is to be added
        :type segment_data: reasonSummary
        """        
        self.actualTimeTaken += segment_data.actualTimeTaken
        self.expectedTimeTaken += segment_data.expectedTimeTaken
        self.opportunity += segment_data.opportunity
        for children in segment_data.children.values():
            self.children.setdefault(segment_data.parentId, []).append(children)
        self.count += segment_data.count
        self.buckets.extend(segment_data.buckets)

    def addCycleData(self, cycle_data):
        """adds cycle reason summary data to machine reason summary

        :param cycle_data: reason summary data of cycle whose data to be added
        :type cycle_data: reasonSummary
        """        
        self.actualTimeTaken += cycle_data.actualTimeTaken
        self.expectedTimeTaken += cycle_data.expectedTimeTaken
        self.opportunity += cycle_data.opportunity
        for children in cycle_data.children.values():
            self.children.setdefault(cycle_data.parentId, []).extend(children)
        self.count += cycle_data.count
        self.buckets.extend(cycle_data.buckets)

    def addMachineData(self, machine_data):
        """adds machines reason summary to find entire datasets reason summary

        :param machine_data: reason summary data of machine whose data to be added
        :type machine_data: reasonSummary
        """        
        self.actualTimeTaken += machine_data.actualTimeTaken
        self.expectedTimeTaken += machine_data.expectedTimeTaken
        self.opportunity += machine_data.opportunity
        for children in machine_data.children.values():
            self.children.setdefault(machine_data.parentId, []).extend(children)
        self.count += machine_data.count
        self.buckets.extend(machine_data.buckets)

    def updateMapProperties(self):
        """this functions calculates and updates loss, efficiency and painIndex of reason summary object
        """        
        if self.actualTimeTaken == 0:
            pass
        else:
            self.loss = self.actualTimeTaken - self.expectedTimeTaken
            self.efficiency = (self.expectedTimeTaken / (self.loss + self.expectedTimeTaken)) * 100
            self.painIndex = 1 - (self.efficiency / 100)

            # for child_list in self.children.values():
            #     self.count += len(child_list)

    def updateToQueuingReason(self):
        """this functions updates aslr reason of reason summary to queuing. This is used when following site
        awareness is occurred in queue area and we want to assign the loss to queuing
        """        
        self.actualAslr = 254
        self.actualAslrName = reasonMap[self.actualAslr]["Reason"]
        self.actualAslrCategory = reasonMap[self.actualAslr]["Category"]

    def getIndexes(self):
        """_summary_
        """        
        indexes = {}
        for machine, machine_data in self.children.items():
            # indexes.setdefault(machine, []).extend([for cycle_data in machine_data] for )
            pass

    @staticmethod
    def addSummaries(summaries, unique_reasons, summary_type, parent_id):
        """this static method adds multiple reason summaries together to get combined summary

        :param summaries: list of reason summaries whose combined summary need to be computed
        :type summaries: list
        :param unique_reasons: list of unique reasons present in above list of reason summaries
        :type unique_reasons: list
        :param summary_type: defines whose summaries we are adding up
        :type summary_type: str
        :param parent_id: id of parent whose summary we are computing
        :type parent_id: str/int
        :return: dictionary containing reason summaries of various aslr reasons in parent
        :rtype: dict
        """        
        output = {}
        for reason in unique_reasons:
            for summary in summaries:
                data = summary.get(reason)
                if data is not None:
                    if reason in output.keys():
                        if summary_type == 'segments':
                            output[reason].addSegmentData(data)
                        elif summary_type == 'cycles':
                            output[reason].addCycleData(data)
                        elif summary_type == 'machines':
                            output[reason].addMachineData(data)
                    else:
                        output[reason] = reasonSummary(data.actualAslr, data.actualAslrName, data.actualAslrCategory,
                                                       parent_id)
                        if summary_type == 'segments':
                            output[reason].addSegmentData(data)
                        elif summary_type == 'cycles':
                            output[reason].addCycleData(data)
                        elif summary_type == 'machines':
                            output[reason].addMachineData(data)

        return output

    def to_dict(self):
        """returns summary of reason

        :return: dictionary containing loss, efficiency, pain Index, count etc due to particular reason
        :rtype: dict
        """        
        return {
            'actualAslr': self.actualAslr,
            'actualAslrName': self.actualAslrName,
            'actualAslrCategory': self.actualAslrCategory,
            'actualTimeTaken (secs)': round(self.actualTimeTaken, 2),
            'expectedTimeTaken (secs)': round(self.expectedTimeTaken, 2),
            'loss (secs)': round(self.loss, 2) if self.loss else 0,
            'efficiency': round(self.efficiency, 2)if self.efficiency else 0,
            'painIndex': round(self.painIndex, 2)if self.painIndex else 0,
            'count': self.count,
        }
