# 56, 57 Merge Intervals & Insert Intervals

## Problem Statement
---
> Problem I: Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.
> 
> Problem II: You are given an array of non-overlapping intervals intervals where intervals[i] = [start_i, end_i] represent the start and the end of the ith interval and intervals is sorted in ascending order by start_i. You are also given an interval newInterval = [start, end] that represents the start and end of another interval.
> Insert newInterval into intervals such that intervals is still sorted in ascending order by starti and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).
> Return intervals after the insertion.
---

## Thought Process

I've merged these two problems together since you use merge intervals to solve Insert Interval. I'll also not being using the slightly-optimized version on leetcode, where you directly handle the case where a given interval is actually completely contained within another. The optimized version notices this and skips the creation of a new interval (which is the same as the outer interval), whereas the code below creates a new interval regardless. It actually makes a noticeable but not significant difference in runtime on leetcode, but the complexity is the same, so whatevs. 

First is Merge Intervals. You could abstract this further with two helper functions (like we do for Insert Interval), but for this part we can write out the entire thing. A noticeable distinction between these two problems is that Merge Intervals is not sorted and can contain several overlapping intervals, while the interval input for Insert Interval is given in a sorted, non-overlapping format.

Okay, so for Merge, we want to sort the input array, which should be a kneejerk reaction for interval problems. Rarerly, you'll encounter an interval problem that doesn't necessitate at least 1 sort. Then, all we have to do is iterate through the intervals and check it against the previous interval. 

There's a specific formula for this, the minimum of the two intervals' ends minus the maximum of the starts. If this value is greater than or equal to 0 (or equal to because start times are inclusive and end-times are exclusive), then we have an overlapping interval. Merging the two is a reversal of the previous formula; now it's the minimum of the start times and the max of the ends, which will give us the outer bounds of the two intervals together.

That's it for Merge, we can append that to a result array as we go. Insert is going to use merge, but we're going to abstract it to make the code clean and reusable. 

We want 3 functions for 3 distinct goals: checking if two intervals overlap, merging only two intervals, and one to merge a list of intervals (so the main function we just wrote). We already went over the two formulas for the two helpers, so this step should be simple af.

For the actual insert capability, we can choose to approach this linearly or using a binary search. Both have the same time complexity (since the upper bound is dictated by the sort), but best practice would be to go with binary search anyways since logN beats N. Create a binary search helper that comparts the start of the new interval to the starts of the intervals list. We want to find the leftmost insertion point; the end times don't matter since we'll be merging the intervals anyways.

With all these helpers, now it's just 3 short lines of code: call the insert function (to put the interval in the list), call the merge_all function, and return.

A confusion I had was why don't we take advantage of the fact that we're given the intervals in non-overlapping, ascending order and only merge the interval that we're inserting with the adjacent interval. If our new interval covers more than just the next interval (or the previous one), it would just create more complexity and essentially be the same as creating a whole new array and checking to see if we can merge intervals one by one (even tho we know that we're not going to find one to merge at least until we hit our insert index).

The time complexity for both is O(NlogN), dominated by the sorting and the followed by several different linear operations. THe space complexity is slightly more tricky. Sorting itself (using TimSort in python) takes logN space, and if we are allowed to modify the input, we can sort in place; otherwise we'd need N linear space to store the sorted array. So for Merge, you can argue either O(logN) or O(N) depending on the implementation. 

The space complexity for Insert is trickier, given our version of it. Despite the fact that we create new arrays during the merging and when we insert our interval, these are always the result array. And, as opposed to the Merge problem, here we didn't have to sort our input. So actually, this takes constant space!

TL;DR Time: O(NlogN), Space-Merge: O(logN) or O(N) depending on if we modify the input, Space-Insert: O(1)

```
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals = sorted(intervals)
        ans = []

        for interval in intervals:
            if ans and min(interval[1], ans[-1][1]) >= max(interval[0], ans[-1][0]):
                ans[-1][0] = min(ans[-1][0], interval[0])
                ans[-1][1] = max(ans[-1][1], interval[1])
            else: 
                ans.append(interval)
        return ans

class Solution:
    def insert(self, intervals: List[List[int]], new_interval: List[int]) -> List[List[int]]:
        def does_overlap(first: List[int], sec: List[int]) -> bool:
            return min(first[1], sec[1]) >= max(first[0], sec[0])
        
        def merge_two(first: List[int], second: List[int]) -> List[int]:
            return [min(first[0], second[0]), max(first[1], second[1])]
        
        def merge_all(intervals: List[List[int]]) -> List[int]:
            result = []
            for interval in intervals:
                if result and does_overlap(interval, result[-1]):
                    result[-1] = merge_two(interval, result[-1])
                else: result.append(interval)
            return result
        
        def insert_linear(new: List[int], intervals: List[List[int]]) -> List[List[int]]:
            for i in range(len(intervals)):
                if new[0] <= intervals[i][0]:
                    intervals.insert(i, new)
                    break
            else:
                intervals.append(new)
            return intervals
        
        def insert_binary(new: List[int], intervals: List[List[int]]) -> List[List[int]]:
            left, right = 0, len(intervals) - 1
            while left <= right:
                mid = left + (right - left) // 2
                if new[0] <= intervals[mid][0]:
                    right = mid - 1
                else: left = mid + 1
            intervals.insert(left, new)
            return intervals
            
        intervals = insert_binary(new_interval, intervals)
        intervals = merge_all(intervals)
        return intervals
```

# 435, Non-overlapping Intervals

## Problem Statement
---
> Given an array of intervals intervals where intervals[i] = [starti, endi], return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.
---

## Thought Process

Sorting as usual for interval problems, but this one is going to be sorted according to our END times not our start times. This just saves us an extra conditional. Then we actually only want to keep track of a main end-point, initialized to something super low (-inf).

When we iterate through our sorted intervals, we only need to check if the start time on the encountered interval is less than our main end. If it is, we're going to discard it and increment our counter by 1. We can do this because we're approaching the problem in a gredy manner.

We know our current endpoint is <= the endpoint of our encountered interval, since we sorted according to end times. And we want to prioritize keeping intervals that end sooner, since (in simplest terms) that mans we're free at a sooner point and can accept more intervals again.

If we're not incrementing our counter, that means we've hit a start time that's >= our main end point. Then, we just have to change our main end to the current interval's end.

The Time complexity is O(NlogN), dominated by the sort at the beginning and followed by a linear run through all the intervals. The space complexity is O(logN) or O(N), depending on if you sorted in-place; that's the only space we need tho, other than two variables.

```
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals = sorted(intervals, key = lambda x: x[1])
        erased, end = 0, -inf

        for interval in intervals:
            if interval[0] < end: erased += 1
            else: end = interval[1]
        
        return erased
```

# 252, 253, 2402 Meeting Rooms I, II, III
## Problem Statement
---
> Problem I: Given an array of meeting time intervals where intervals[i] = [start_i, end_i], determine if a person could attend all meetings.
> 
> Problem II: Given an array of meeting time intervals intervals where intervals[i] = [start_i, end_i], return the minimum number of conference rooms required.
>
> Problem III: You are given an integer n. There are n rooms numbered from 0 to n - 1.
> You are given a 2D integer array meetings where meetings[i] = [starti, endi] means that a meeting will be held during the half-closed time interval [starti, endi). All the values of starti are unique. Meetings are allocated to rooms in the following manner:
> Each meeting will take place in the unused room with the lowest number.
If there are no available rooms, the meeting will be delayed until a room becomes free. The delayed meeting should have the same duration as the original meeting.
When a room becomes unused, meetings that have an earlier original start time should be given the room.
> Return the number of the room that held the most meetings. If there are multiple rooms, return the room with the lowest number.
> A half-closed interval [a, b) is the interval between a and b including a and not including b.
---

## Thought Process

An entire family of problems. The first one is super easy given our knowledge of intervals from the previous problems. 

We have to sort as usual, but then we iterate through the array and check to see if the start time of the current interval is less than the end time of the previous interval. If it is, we return False. Super quick, it's done in O(NlogN) time and O(N) or O(logN) space. Even though it's a linear search, we still have to sort the input intervals, and the space required is O(logN) minimum for sorting, and O(N) to stored our sorted array if we don't sort in-place, which we shouldn't.

The second one is a bit more interesting and can be performed 2 separate wyas: chronological ordering or heaps. The chronological method is actually slightly less efficient on Leetcode despite having the same time complexity, but we'll get into that later.

Basically, for chronological, we want to create two different sorted arrays for just the start times and the end times. then we use a two-pointer iteration method to go through primarily the start array (since that's our limiting element). If we have a meeting that needs to start and it's greate then where we're at in the end array, then we need a new room. However, if it's less than or equal to the end time, then we can just increment the end index as well as our usual start index. We can do this specifically because it doesn't matter which meeting ended, it just matters that one did, so the specific ordering doesn't matter as much.

The heap method is shorter and simpler. Sort the intervals according to start time, and then we need three variables: an int to count our rooms, an int to represent our available rooms, and an empty heap for our occupied rooms.

We start by popping off the heap if the top value is less than or equal to the start of our current interval. This means a meeting ended, so we've freed up a room and should increment our available counter as well. 

Now if we still don't have an available room, that means we weren't able to pop any off, so we should add a new room; if we had a room, we'd just decrement the counter. Finally in the loop, we push the end of our current interval, easy as pie.

The time complexity is O(NlogN) for both methods. The chrono method is about O(2NlogN + N), and the heap caps at O(3NlogN) but tends to run faster on Leetcode. The space complexity is the same as well, O(N) for both.

Now, the last one, a hard problem that you shoul be able to do easily after everything. It functions using 2 heaps: one for available rooms, and one for occupied rooms. The occupied heap is ordered by the end time, which includes the delay, and also contains the room used. The available heap only containts the room ints. We can use a dictionary or an array to act as a counter for the rooms as we "book" them. 

Going through a sorted version of the given input array, we pop off elements of the occupied heap and push those rooms back onto the available heap. Then it's a simple if-else statement to see if there's a room available or not. If there isn't, we have to calculate a delay time until the next room opens up, and pop off that next room from the occupied heap.

After the loop we need to find the INDEX of the max count, which can be easily done manually if we can't use the built-in index function. 

The time complexity is O(NlogN) for sorting and heap operations, and the space complexity is O(N) for the 2 heaps, the counter array or dictionary, and the sorted array.

```
class Solution:
    # T: O(NlogN)  S: O(N) or O(logN)
    def meetings_I(self, intervals: List[List[int]]) -> bool:
        intervals = sorted(intervals)
        for i in range(1, len(intervals)):
            if intervals[i][0] < intervals[i - 1][1]: return False
        return True

    # T: O(NlogN) (but always 2NlogN + N)  S: O(N) or O(logN) 
    def meetings_II_chrono(self, intervals: List[List[int]]) -> int:
        starts = sorted([i[0] for i in intervals])
        ends = sorted([i[1] for i in intervals])

        s_ind = e_ind = min_rooms = 0
        while s_ind < len(intervals):
            if starts[s_ind] < ends[e_ind]: 
                min_rooms +=  1
            else:
                e_ind += 1
            s_ind += 1

        return min_rooms

        intervals = sorted(intervals)
        max_rooms, available, occupied = 0, 0, []

    # T: O(NlogN) (but maxes at 3NlogN) S: O(N) or O(logN)
    def meetings_II_heap(self, intervals: List[List[int]]) -> int:
        intervals = sorted(intervals)
        max_rooms, available, occupied = 0, 0, []

        for interval in intervals:
            while occupied and interval[0] >= occupied[0]:
                heappop(occupied); available += 1
            
            if available == 0: max_rooms += 1
            else: available -= 1
            heappush(occupied, interval[1])
        return max_rooms

    # T: O(NlogN) S: O(N)
    def meetings_III(self, n: int, meetings: List[List[int]]) -> int:
        available, occupied = [x for x in range(n)], []
        count = [0] * n

        for start, end in sorted(meetings):
            while occupied and start >= occupied[0][0]:
                e, room = heappop(occupied)
                heappush(available, room)
            
            delay = 0
            if available:
                room = heappop(available)
            else:
                delay, room = heappop(occupied)
                delay -= start
            heappush(occupied, [end + delay, room])
            count[room] += 1
        
        return count.index(max(count))
```

# 1851, Minimum Interval to Include Each Query

## Problem Statement
---
> You are given a 2D integer array intervals, where intervals[i] = [left_i, right_i] describes the ith interval starting at left_i and ending at right_i (inclusive). The size of an interval is defined as the number of integers it contains, or more formally right_i - left_ + 1.
> You are also given an integer array queries. The answer to the jth query is the size of the smallest interval i such that left_i <= queries[j] <= right_i. If no such interval exists, the answer is -1.
> Return an array containing the answers to the queries.
---

## Thought Process

Bit tricky even after solving other interval problems, but the biggest help here is to give up on maintaining the order of queries without additional help (in this example, we used dictionaries). Might be a more efficient way but I can't find one yet. Once we treat the queries and intervals separately, this becomes a lot easier.

We're using heaps and sorts, looping through sorted queries. We're given the condition that we need to satisfy for the query, so we can first start by checking for intervals with a start time less than or equal to our query. We don't want to add to the heap yet, but once we confirm this start time <=, we'll want to move to the next index regardless of whether or not the end time is before or after the query. We'll only add to the heap when query is less than or equal to the end of the current interval.

The heap should be ordered by size, so in the form [size, end] which can be further defined as [end - start + 1, end]. 

Now since we have all the viable intervals in our heap (since intervals were ordered by the start times, so no further intervals would work for a given query) we want to pop off the heap while the query is greather than the end time in the heap. That's because we structured our queries to be in non-decreasing order as well, so if the current q is greater than the end time, none of the following queries would be able to use that interval anyways. 

Now we can finally use our dictionary to store the size of the top of the heap with the query as a key.

And finally, we can use a list comprehensions to return the result in the given input order of the queries, rather than our sorted order.

The Time complexity is dictated by the sort, as usual. There's also the logN heap operations, but we'll at most perform the pushing and popping N times, so those maintain the NlogN time complexity. Reformatting the result is done in linear time. 

The space complexity is O(N), to store the result dictionary and the heap. There's also the minimum logN space needed to perform a sort operation. 

```
from heapq import heappop, heappush
class Solution:
    def minInterval(self, intervals: List[List[int]], queries: List[int]) -> List[int]:
        intervals = sorted(intervals)
        heap, result = [], {}
        ind = 0

        for q in sorted(queries):
            while ind < len(intervals) and q >= intervals[ind][0]:
                if q <= intervals[ind][1]: 
                    heappush(heap, [intervals[ind][1] - intervals[ind][0] + 1, intervals[ind][1]])
                ind += 1
            while heap and q > heap[0][1]: heappop(heap)
            result[q] = heap[0][0] if heap else -1
        
        return [result[q] for q in queries]
```
