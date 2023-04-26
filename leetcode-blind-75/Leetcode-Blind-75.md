
- [**Hashmap \& Array**](#hashmap--array)
  - [**Easy**](#easy)
  - [**Medium**](#medium)
- [**Two Pointers**](#two-pointers)
  - [**Easy**](#easy-1)
  - [**Medium**](#medium-1)
  - [Hard](#hard)
- [**Sliding window**](#sliding-window)
  - [**Easy**](#easy-2)
  - [**Medium**](#medium-2)
  - [Hard](#hard-1)
- [**Stack**](#stack)
  - [**Easy**](#easy-3)
  - [**Medium**](#medium-3)
  - [Hard](#hard-2)
- [**Binary Search**](#binary-search)
  - [**Easy**](#easy-4)
  - [**Medium**](#medium-4)
  - [Hard](#hard-3)
- [**Linked List**](#linked-list)
  - [**Easy**](#easy-5)
  - [**Medium**](#medium-5)
  - [Hard](#hard-4)
- [**Trees**](#trees)
  - [**Easy**](#easy-6)
  - [**Medium**](#medium-6)
  - [Hard](#hard-5)
- [Tries](#tries)
  - [Medium](#medium-7)
  - [Hard](#hard-6)
- [**Heap/ Priority Queue**](#heap-priority-queue)
  - [**easy**](#easy-7)
  - [Medium](#medium-8)
  - [Hard](#hard-7)
- [Backtracking](#backtracking)
  - [Medium](#medium-9)
  - [Hard](#hard-8)
- [Graph](#graph)
  - [Medium](#medium-10)
  - [Hard](#hard-9)
- [Advanced Graphs](#advanced-graphs)
  - [Medium](#medium-11)
  - [Hard](#hard-10)
- [**1-D Dynamic Programming**](#1-d-dynamic-programming)
  - [E**asy**](#easy-8)
  - [Medium](#medium-12)
- [2-D Dynamic Programming](#2-d-dynamic-programming)
  - [Medium](#medium-13)
  - [Hard](#hard-11)
- [**Greedy**](#greedy)
  - [**Easy**](#easy-9)
  - [Medium](#medium-14)
- [**Intervals**](#intervals)
  - [**Easy**](#easy-10)
  - [Medium](#medium-15)
  - [Hard](#hard-12)
- [**Math \& Geometry**](#math--geometry)
  - [**Easy**](#easy-11)
  - [Medium](#medium-16)
- [**Bit Manipulation 位运算**](#bit-manipulation-位运算)
  - [**Easy**](#easy-12)


# **Hashmap & Array**
## **Easy** 


- **[217.Contains Duplicate](https://leetcode.com/problems/contains-duplicate/)**
    
    ### **Time and Space Complexity**
    
    | Method | TIme | Space |
    | --- | --- | --- |
    | loop through the array to compare each element | O(n^2)
    (n is the size of the input array) | O(1) |
    | sorting: any duplicates exists in the array would be in adjacent, comparing 2 neighbors in the array | O(nlogn)
    sorting would take extra time | O(1) |
    | use HashSet (Use extra memory): ask the hashset if a certain value exists | O(n) | O(n)
    memory could up to the size of the input array |
    
    ### **Solution**
    
    ```python
    class Solution:
        def containsDuplicate(self, nums: List[int]) -> bool:
            hashset = set() # create the hashset
    
            for num in nums:    # going through every value in the array
                if num in hashset: # check if the number already in the hashset
                    return True
                # if the hashset doesn't contain the value, add the number
                hashset.add(num)
            return False
    
    ```


- **[242.Valid Anagram](https://leetcode.com/problems/valid-anagram/)**
    
    #### **Solution 1**
    
    ```python
    class Solution:
        def isAnagram(self, s: str, t: str) -> bool:
            if len(s) != len(t):
                return False
    
            countS, countT = {}, {}
    
            # building two hashmaps
            for i in range(len(s)):
                countS[s[i]] = countS.get(s[i], 0) + 1
                countT[t[i]] = countT.get(t[i], 0) + 1
            for c in countS:    # iterate the key value
                if countS[c] != countT.get(c, 0):  # the key could also not exist in the t array
                    return False
    
            return True
    ```
    
    Time: O(s+t); Space: O(s+t)
    
    ### **Solution 2**
    
    If you are asked to provide a solution without extra sapce:
    
    the letters would be the same string in the same sorted order, downside is the time complexity O(n_2)orO(nlog(n)) ; Space complexity might be O(1)-->(usually the what the interviewer think of the space complexity for sorting algorithm, could discuss with the interviewer)
    
    ```
    class Solution:
        def isAnagram(self, s: str, t: str) -> bool:
            return sorted(s) == sorted(t)
    ```
    
    - Interviewer may ask to write your own sorted function.
- **[1.Tow Sum](https://leetcode.com/problems/two-sum/)**
    - the most popular leetcode question
    
    time complexity O(n): only need to iterate the array once
    
    space complexity  O(n): need extra space to store the hashmap
    
    ### **Solution**
    
    ```python
    class Solution:
        def twoSum(self, nums: List[int], target: int) -> List[int]:
            prevMap = {}    # store every previous value
    
            for i, n in enumerate(nums):
                diff = target - n
                if diff in prevMap:
                    return [prevMap[diff], i]
                prevMap[n] = i
            return
    ```
    

## **Medium**

- **[49. Group Anagrams](https://leetcode.com/problems/group-anagrams/)**
    
    ```python
     class Solution:
         def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
             # t-O(m·n), m is the length of the list, n is the avg length of each character
    
             # use the defaultdict to prevent edge cases
             # eg. when you search for a non-exist key,
             # defaultdict(list) would return []
             # defaultdict(int) would return 0
             # defaultdict(set) would return set()
             # defaultdict(str) would return
             res = defaultdict(list)
    
             for s in strs:
                 # iterate each s in the string list
                 count = [0] * 26    # use count to store the appearance time of 26 alphabetical nums
                 # initial value is 0
    
                 for c in s:
                     # update the count for each character
                     count[ord(c) - ord('a')] += 1   # use ASCII to get the correct idx
    
                 # python do not support using list as the key, use tuple() instead
                 res[tuple(count)].append(s)
    
             return res.values()
    ```
    
- **[347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)**
    
    ```python
    class Solution:
        def topKFrequent(self, nums: List[int], k: int) -> List[int]:
            # t-O(n), m-O(n)
    
            count = {}  # a hashmap to count the occurance of each number
            freq = [[] for i in range(len(nums)+1)]  # an array with key-count and value-number
    
            # update the hashmap
            for num in nums:
                count[num] = count.get(num, 0) + 1
    
            # update the freq array based on the hashmap
            for num, occ in count.items():
                freq[occ].append(num)
    
            # loop from the back of the freq array to find top k nums
            res = []
            for count in range(len(freq)-1, 0, -1):
                for num in freq[count]:
                    res.append(num)
                    if len(res) == k:
                        return res
    ```
    
- **[238. Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/)**
    
    ```python
    class Solution:
        def productExceptSelf(self, nums: List[int]) -> List[int]:
            res = [1] * (len(nums))
    
            prefix = 1
            for i in range(len(nums)):
                res[i] = prefix
                # update prefix
                prefix *= nums[i]
    
            postfix = 1
            for i in range(len(nums)-1, -1, -1):
                res[i] *= postfix
                postfix *= nums[i]
            return res
    ```
    
- **[36. Valid Sudoku](https://leetcode.com/problems/valid-sudoku/)**
    
    ```python
    class Solution:
        def isValidSudoku(self, board: List[List[str]]) -> bool:
            col = defaultdict(set)
            row = defaultdict(set)
            square = defaultdict(set)
    
            for c in range(9):
                for r in range(9):
                    if board[c][r] == '.':
                        continue
                    if ((board[c][r] in col[c]) or
                       (board[c][r] in row[r]) or
                       (board[c][r] in square[c //3, r //3])):
                        return False
    
                    col[c].add(board[c][r])
                    row[r].add(board[c][r])
                    square[c //3, r //3].add(board[c][r])
    
            return True
    ```
    
- **[659 · Encode and Decode Strings](https://www.lintcode.com/problem/659/)**
    
    ```
    class Solution:
        """
        @param: strs: a list of strings
        @return: encodes a list of strings to a single string.
        """
        def encode(self, strs):
            # write your code here
            res = ""
            for s in strs:
                res += str(len(s)) + "#" + s
    
            return res
    
        """
        @param: str: A string
        @return: dcodes a single string to a list of strings
        """
        def decode(self, str):
            # write your code here
            res = []
            i = 0
    
            while i < len(str):
                j = i
                while str[j] != "#":
                    j += 1
                length = int(str[i: j])
                res.append(str[j+1 : j+1+length])
                i = j + 1 + length
    
            return res
    
    ```
    
- **[128. Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)**
    
    ```python
    class Solution:
        def longestConsecutive(self, nums: List[int]) -> int:
            numSet = set(nums)
            longest = 0
    
            for num in nums:
                if (num-1) not in numSet:
                    length = 0
                    while (num + length) in numSet:
                        length += 1
                    longest = max(longest, length)
            return longest
    ```
    

# **Two Pointers**

## **Easy**

- **[125.Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)**
    - Compare the left and right character using two pointers.
    - Stop point: two pointers meet in the middle, or they passed each other.
    - Using ASCII value to determine if the characters is alphanumerical.
    
    time: O(n)--iterate through the string
    
    space: O(1)--no extra memory
    
    ```python
    class Solution:
        def isPalindrome(self, s: str) -> bool:
    
            # set 2 pointers
            l, r = 0, len(s)-1
    
            while l < r:
                # make sure both left and right are alphanumeric
                while l < r and not self.alphNum(s[l]):
                    l += 1
                while l < r and not self.alphNum(s[r]):
                    r -= 1
    
                if s[l].lower() != s[r].lower():
                    return False
    
                l += 1
                r -= 1
    
            return True
    
        # define for alphanumerica characters
        def alphNum(self, c):
            return (ord('A') <= ord(c) <= ord('Z') or
            ord('a') <= ord(c) <= ord('z') or
            ord('0') <= ord(c) <= ord('9'))
    ```
    

## **Medium**

- **[167. Two Sum II - Input Array Is Sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)**
    
    ```python
    class Solution:
        def twoSum(self, numbers: List[int], target: int) -> List[int]:
            l, r = 0, len(numbers)-1
    
            while l < r:
                if numbers[l] + numbers[r] < target:
                    l += 1
                elif numbers[l] + numbers[r] > target:
                    r -= 1
                if numbers[l] + numbers[r] == target:
                    return [l+1, r+1]
    
    ```
    
- **[15. 3Sum](https://leetcode.com/problems/3sum/)**
    
    ```python
    class Solution:
        def threeSum(self, nums: List[int]) -> List[List[int]]:
            res = []
            # sort the nums
            nums.sort()
    
            for i in range(len(nums)):
                if i > 0 and nums[i] == nums[i-1]:
                    continue
    
                # implement two sum II
                l = i + 1
                r = len(nums)-1
    
                while l < r:
                    if nums[i] + nums[l] + nums[r] > 0:
                        r -= 1
                    elif nums[i] + nums[l] + nums[r] < 0:
                        l += 1
                    else:
                        res.append([nums[i], nums[l], nums[r]])
                        l += 1
                        while nums[l] == nums[l-1] and l < r:
                            l += 1
    
            return res
    ```
    
- **[11. Container With Most Water](https://leetcode.com/problems/container-with-most-water/)**
    
    ```python
    class Solution:
        def maxArea(self, height: List[int]) -> int:
            l, r = 0, len(height)-1
            maxA = 0
    
            while l < r:
                curA = (r-l) * min(height[l], height[r])
                maxA = max(curA, maxA)
                if height[l] < height[r]:
                    l += 1
                else:
                    r -= 1
            return maxA
    ```
    

## Hard

- **[42. Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)**
    
    Given `n` non-negative integers representing an elevation map where the width of each bar is `1`, compute how much water it can trap after raining.
    
    **Example 1:**
    
    ![https://assets.leetcode.com/uploads/2018/10/22/rainwatertrap.png](https://assets.leetcode.com/uploads/2018/10/22/rainwatertrap.png)
    
    ```
    Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
    Output: 6
    Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.
    
    ```
    
    **Example 2:**
    
    ```
    Input: height = [4,2,0,3,2,5]
    Output: 9
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled.png)
    
    ```python
    class Solution:
        def trap(self, height: List[int]) -> int:
            if not height:
                return 0
            
            l, r = 0, len(height) - 1
            leftMax, rightMax = height[0], height[-1]
            res = 0
            
            while l < r:
                if leftMax < rightMax:
                    l += 1
                    if leftMax - height[l] >= 0:
                        res += leftMax - height[l]
                    leftMax = max(leftMax, height[l])
                else:
                    r -= 1
                    if rightMax - height[r] >= 0:
                        res += rightMax - height[r]
                    rightMax = max(rightMax, height[r])
            
            return res
    ```
    

# **Sliding window**

## **Easy**

- **[121.Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)**
    - stock market: buy low and sell high
    - want an transaction that maximize the profit
    
    time: O(n)
    
    space: O(1)
    
    ```python
    class Solution:
        def maxProfit(self, prices: List[int]) -> int:
              ###### my answer:
    #         l, r = 0, 1 # initialize two pointers
    #         maxP = 0
    
    #         while r < len(prices):
    #             Profit = prices[r] - prices[l]
    
    #             if Profit < 0 and l < r:
    #                 l= l+1
    #             else:
    #                 r += 1
    #                 if Profit >= maxP:
    #                     maxP = Profit
    
    #         return maxP
            l, r = 0, 1
            maxP = 0
    
            while r < len(prices):
                # profitable?
                if prices[l] < prices[r]:
                    profit = prices[r] - prices[l]
                    maxP = max(profit, maxP)
                else:
                    l = r   # shift the left pointer right to the right point
                r += 1  # increment the right pointer in all cases
    
            return maxP
    ```
    

## **Medium**

- **[3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)**
    
    ```python
    class Solution:
        def lengthOfLongestSubstring(self, s: str) -> int:
            charSet = set()
            res = 0
            l = 0
    
            for r in range(len(s)):
                while s[r] in charSet:
                    charSet.remove(s[l])
                    l += 1
                charSet.add(s[r])
                res = max(res, r-l+1)
            return res
    ```
    
- **[424. Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/)**
    
    ```python
    class Solution:
        def characterReplacement(self, s: str, k: int) -> int:
            count = {}  # a hashmap to store the occurance of characters-maxlen=26
            res = 0 # initialize the length of the longest substring
    
            l = 0 # left pointer
            for r in range(len(s)):
                count[s[r]] = count.get(s[r], 0) + 1
                # number of strings to be replaced
                while (r-l+1) - max(count.values()) > k:
                    count[s[l]] -= 1
                    l += 1
    
                res = max(res, r-l+1)
            return res
    ```
    
- **[567. Permutation in String](https://leetcode.com/problems/permutation-in-string/)**
    
    ```python
    class Solution:
        def checkInclusion(self, s1: str, s2: str) -> bool:
            if len(s1) > len(s2):
                return False
    
            # create two list for s1 and s2
            s1Count, s2Count = [0] * 26, [0] * 26
            # initialize the lists for both s1 and s2
            for i in range(len(s1)):
                s1Count[ord(s1[i]) - ord('a')] += 1
                s2Count[ord(s2[i]) - ord('a')] += 1
    
            matches = 0
            for i in range(26):
                # the initial matches
                matches += (1 if s1Count[i] == s2Count[i] else 0)
    
            l = 0
            for r in range(len(s1), len(s2)):
                if matches == 26:
                    return True
    
                # modify the current list
                idx_r = ord(s2[r]) - ord('a')
                s2Count[idx_r] += 1
                if s1Count[idx_r] == s2Count[idx_r]:
                    matches += 1
                elif s1Count[idx_r] + 1 == s2Count[idx_r]:
                    matches -= 1
    
                idx_l = ord(s2[l]) - ord('a')
                s2Count[idx_l] -= 1
                if s1Count[idx_l] == s2Count[idx_l]:
                    matches += 1
                elif s1Count[idx_l] -1 == s2Count[idx_l]:
                    matches -= 1
                l += 1
            return matches == 26
    ```
    

## Hard

- **[76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)**
    
    Given two strings `s` and `t` of lengths `m` and `n` respectively, return *the **minimum window substring** of* `s` *such that every character in* `t` *(**including duplicates**) is included in the window. If there is no such substring, return the empty string* `""`*.*
    
    The testcases will be generated such that the answer is **unique**.
    
    A **substring** is a contiguous sequence of characters within the string.
    
    **Example 1:**
    
    ```
    Input: s = "ADOBECODEBANC", t = "ABC"
    Output: "BANC"
    Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.
    
    ```
    
    **Example 2:**
    
    ```
    Input: s = "a", t = "a"
    Output: "a"
    Explanation: The entire string s is the minimum window.
    
    ```
    
    **Example 3:**
    
    ```
    Input: s = "a", t = "aa"
    Output: ""
    Explanation: Both 'a's from t must be included in the window.
    Since the largest window of s only has one 'a', return empty string.
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%201.png)
    
    ```python
    class Solution:
        def minWindow(self, s: str, t: str) -> str:
            if t == "":
                return ""
            
            countMap, targetMap = {}, {}
             # create the targetMap
            for c in t:
                targetMap[c] = targetMap.get(c, 0) + 1
            
            have, need = 0, len(targetMap)
            res = [-1, -1] # start and end idx
            minLen = float("infinity")
            l = 0
            
            for r in range(len(s)):
                c = s[r]
                countMap[c] = countMap.get(c, 0) + 1
                
                if c in targetMap and countMap[c] == targetMap[c]:
                    have += 1
                
                while have == need:
                    # compare the length
                    if (r - l + 1) < minLen:
                        res = [l, r]
                        minLen = (r - l + 1)
                    
                    # pop from the left of the current window
                    countMap[s[l]] -= 1
                    # if the poped value in target
                    if s[l] in targetMap and countMap[s[l]] < targetMap[s[l]]:
                        have -= 1
                    l += 1
            
            l, r = res
            return s[l : r + 1] if minLen != float("infinity") else ""
    ```
    
- **[239. Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)**
    
    You are given an array of integers `nums`, there is a sliding window of size `k` which is moving from the very left of the array to the very right. You can only see the `k` numbers in the window. Each time the sliding window moves right by one position.
    
    Return *the max sliding window*.
    
    **Example 1:**
    
    ```
    Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
    Output: [3,3,5,5,6,7]
    Explanation:
    Window position                Max
    ---------------               -----
    [1  3  -1] -3  5  3  6  73
     1 [3  -1  -3] 5  3  6  73
     1  3 [-1  -3  5] 3  6  7 5
     1  3  -1 [-3  5  3] 6  75
     1  3  -1  -3 [5  3  6] 76
     1  3  -1  -3  5 [3  6  7]7
    ```
    
    **Example 2:**
    
    ```
    Input: nums = [1], k = 1
    Output: [1]
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%202.png)
    
    ```python
    class Solution:
        def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
            output = []
            q = collections.deque() # store idx
            l = r = 0
            
            while r < len(nums):
                # pop smaller values from r
                while q and nums[q[-1]] < nums[r]:
                    q.pop()
                q.append(r)
                
                # remove left val from window (out of bounds)
                if l > q[0]:
                    q.popleft()
                
                # append the max value for one window
                if (r + 1) >= k:
                    output.append(nums[q[0]])
                    l += 1
                r += 1
            
            return output
    ```
    

# **Stack**

## **Easy**

- **[20.Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)**
    - a very common interview question
    
    time: O(n)
    
    space: O(n)
    
    ```python
    class Solution:
        def isValid(self, s: str) -> bool:
            stack = []
            # hashmap
            closToOpen = {
                ')': '(',
                '}': '{',
                ']': '['
            }
    
            parStack = []
    
            for c in s:
                if c in closToOpen:
                    if stack and stack[-1] == closToOpen[c]:
                        stack.pop()
                    else:
                        return False
                else:   # get the open parentheses
                    stack.append(c)
    
            return True if not stack else False
    ```
    
- **[155.Min stack](notion://www.notion.so/leahishere/155/Min%20stack)**
    - use 2 stacks, one to do the top/pop/push function, another responsible for the getMin function.
    
    ```python
    class MinStack:
    
        def __init__(self):
            self.stack = []
            self.minStack = []
    
        def push(self, val: int) -> None:
            self.stack.append(val)
            val = min(val, self.minStack[-1] if self.minStack else val)
            # get the minimum value of the current input value and the top value in minStack
            # if minStack is empty, val is the current val
            self.minStack.append(val)
    
        def pop(self) -> None:
            self.stack.pop()
            self.minStack.pop()
    
        def top(self) -> int:
            return self.stack[-1]
    
        def getMin(self) -> int:
            return self.minStack[-1]
    
    # Your MinStack object will be instantiated and called as such:
    # obj = MinStack()
    # obj.push(val)
    # obj.pop()
    # param_3 = obj.top()
    # param_4 = obj.getMin()
    ```
    

## **Medium**

- **[150. Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/)**
    
    ```python
    class Solution:
        def evalRPN(self, tokens: List[str]) -> int:
            res = []
    
            for c in tokens:
                if c == "+":
                    num1 = res.pop()
                    num2 = res.pop()
                    res.append(num1 + num2)
                elif c == "-":
                    num1 = res.pop()
                    num2 = res.pop()
                    res.append(num2 - num1)
                elif c == "*":
                    num1 = res.pop()
                    num2 = res.pop()
                    res.append(num1 * num2)
                elif c == "/":
                    num1 = res.pop()
                    num2 = res.pop()
                    res.append(int(num2/num1))	# 注意此处的取整操作
                else:
                    res.append(int(c))
            return res[0]
    ```
    
- **[22. Generate Parentheses](https://leetcode.com/problems/generate-parentheses/)**
    
    ```python
    class Solution:
        def generateParenthesis(self, n: int) -> List[str]:
            # open number = n
            # close number = n
            # only add open if open < n
            # only add closing if closing < open
            # valid IIF open == closed = n
    
            stack = []
            res = []
    
            def backtrack(openN, closeN):
                if openN == closeN == n:s
                    res.append("".join(stack))
                    return
    
                if openN < n:
                    stack.append("(")
                    backtrack(openN + 1, closeN)
                    stack.pop( )
    
                if closeN < openN:
                    stack.append(")")
                    backtrack(openN, closeN+1)
                    stack.pop()
    
            backtrack(0, 0)
            return res
    
    ```
    
- **[739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)**
    
    ```python
    class Solution:
        def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
            stack = []  # temperature, idx
            res = [0] * len(temperatures)
    
            for i, t in enumerate(temperatures):
                while stack and t > stack[-1][0]:
                    stackT, stackIdx = stack.pop()
                    res[stackIdx] = i - stackIdx
                stack.append([t, i])
            return res
    ```
    
- **[853. Car Fleet](https://leetcode.com/problems/car-fleet/)**
    
    time: o(nlog(n))
    
    ```python
    class Solution:
        def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
            stack = []
            pairs = [[p, s] for p, s in zip(position, speed)]
    
            for p, s in sorted(pairs)[::-1]:    # in reversed order
                stack.append((target-p)/s)
                if len(stack) > 1 and stack[-1] <= stack[-2]:
                    stack.pop()
            return len(stack)
    ```
    

## Hard

- **[84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)**
    
    Given an array of integers `heights` representing the histogram's bar height where the width of each bar is `1`, return *the area of the largest rectangle in the histogram*.
    
    **Example 1:**
    
    ![https://assets.leetcode.com/uploads/2021/01/04/histogram.jpg](https://assets.leetcode.com/uploads/2021/01/04/histogram.jpg)
    
    ```
    Input: heights = [2,1,5,6,2,3]
    Output: 10
    Explanation: The above is a histogram where width of each bar is 1.
    The largest rectangle is shown in the red area, which has an area = 10 units.
    
    ```
    
    **Example 2:**
    
    ![https://assets.leetcode.com/uploads/2021/01/04/histogram-1.jpg](https://assets.leetcode.com/uploads/2021/01/04/histogram-1.jpg)
    
    ```
    Input: heights = [2,4]
    Output: 4
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%203.png)
    
    ```python
    class Solution:
        def largestRectangleArea(self, heights: List[int]) -> int:
            stack = []
            maxArea = 0
            
            for i, h in enumerate(heights):
                start = i
                # if the height is no longer increasing
                while stack and stack[-1][1] > h:
                    idx, height = stack.pop()
                    maxArea = max(maxArea, height * (i - idx))
                    # the start idx could be extended backwards
                    start = idx
                stack.append((start, h))
                
            # for the remaining pairs in the stack
            for i, h in stack:
                # could be all the way extended to the end
                maxArea = max(maxArea, h * (len(heights) - i))
            
            return maxArea
    ```
    

# **Binary Search**

- **[206.Reversed Linked List](https://leetcode.com/problems/reverse-linked-list/)**
    
    Two ways:
    
    - iteratively: pointers
    
    time: O(n)--just using pointers, no other data structures
    
    space: O(1)
    
    ```python
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    class Solution:
        def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
            prev, curr = None, head
    
            while curr:
                nxt = curr.next
                curr.next = prev
                prev = curr
                curr = nxt
            return prev
    ```
    
    - recursively (hard to understand right now)
    
    time: O(n)
    
    space: O(n)
    
    ```
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    class Solution:
        def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
            if not head:
                return None
    
            newHead = head
            if head.next:
                newHead = self.reverseList(head.next)
                head.next.next = head
            head.next = None
            return newHead
    ```
    

## **Easy**

- **[704/Binary Search](https://leetcode.com/problems/binary-search/)**
    - time complexity O(logn)
        - how many times would divide n by 2? --> base 2
    - should implement very quickly!
    
    ```
    class Solution:
        def search(self, nums: List[int], target: int) -> int:
            l, r = 0, len(nums) - 1
    
            while l <= r:
                m = (l + r) // 2    # may overflow somtimes
                # m = l + (r-l)//2 this version of calculation would never overflow
                if nums[m] > target:
                    r = m - 1
                elif nums[m] < target:
                    l = m + 1
                else:
                    return m
            return -1
    ```
    

## **Medium**

- **[74. Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/)**
    
    ```python
    # my solution
    class Solution:
        def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
            r, c = 0, len(matrix[0]) - 1
    
            while c >= 0 and r < len(matrix):
                if matrix[r][c] < target:
                    r += 1
                elif matrix[r][c] > target:
                    c -= 1
                else:
                    return True
    
            return False
    ```
    
    ```python
    # a more efficient algorithm
    class Solution:
        def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
            # using two binary searches
            ROW, COL = len(matrix), len(matrix[0])
    
            # the first binary search to find rows
            top, bot = 0, ROW -1
            while top <= bot:
                row =  top + (bot - top)//2
                if target > matrix[row][-1]:
                    top = row + 1
                elif target < matrix[row][0]:
                    bot = row - 1
                else:
                    break   # we found the row that the target in
    
            # if the first while loop did not break and
            # the while condition does not satisfied anymore
            if not (top <= bot):
                return False
    
            row =  top + (bot - top)//2
            l, r = 0, COL -1
            while l <= r:
                m = l + (r - l)//2
                if target < matrix[row][m]:
                    r -= 1
                elif target > matrix[row][m]:
                    l += 1
                else:
                    return True
            return False
    ```
    
- **[875. Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/)**
    
    ```python
    class Solution:
        def minEatingSpeed(self, piles: List[int], h: int) -> int:
            mink = max(piles)
            l, r = 1, max(piles)
    
            while l <= r:
                k = l + (r - l)//2
                sumH = 0
                for p in piles:
                    sumH += ((p-1)//k + 1)	###### notice here
                if sumH <= h:
                    mink = min(mink, k)
                    r = k - 1
                else:
                    l = k + 1
    
            return mink
    ```
    
- **[33. Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)**
    
    NOTE: You must write an algorithm with `O(log n)` runtime complexity.
    
    - Means you are possibly finding a solution with binary search.
    
    ```python
    class Solution:
        def search(self, nums: List[int], target: int) -> int:
            l, r = 0, len(nums)-1
            
            while l <= r:
                m = l + (r - l)//2
                if nums[m] == target:
                    return m
                
                # left portion
                if nums[l] <= nums[m]:
                    if target > nums[m] or target < nums[l]:
                        l = m + 1
                    else:
                        r = m - 1
                else:
                    if target < nums[m] or target > nums[r]:
                        r = m - 1
                    else:
                        l = m + 1
            return -1
    ```
    
- **[153. Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)**
    
    ```python
    class Solution:
        def findMin(self, nums: List[int]) -> int:
            res = nums[0]
            l, r = 0, len(nums)-1
    
            while l <= r:
                if nums[l] < nums[r]:
                    res = min(res, nums[l])
                    break
    
                m = l + (r - l )//2
                res = min(res, nums[m])
    
                if nums[m] < nums[l]:
                    r = m - 1
                else:
                    l = m + 1
    
            return res
    ```
    
- **[981. Time Based Key-Value Store](https://leetcode.com/problems/time-based-key-value-store/)**
    
    ```python
    class TimeMap:
    
        def __init__(self):
            self.map = {}   # key: list of [[value, timestamp]]
    
        def set(self, key: str, value: str, timestamp: int) -> None:
            if key not in self.map:
                self.map[key] = []
            self.map[key].append([value, timestamp])
    
        def get(self, key: str, timestamp: int) -> str:
            res = ""
            values = self.map.get(key, [])
    
            l, r = 0, len(values)-1
            while l <= r:
                m = l + (r - l)//2
                if values[m][1] <= timestamp:
                    res = values[m][0]
                    l = m + 1
                else:
                    r = m - 1
    
            return res
    
    # Your TimeMap object will be instantiated and called as such:
    # obj = TimeMap()
    # obj.set(key,value,timestamp)
    # param_2 = obj.get(key,timestamp)
    ```
    

## Hard

- **[4. Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/)**
    
    Given two sorted arrays `nums1` and `nums2` of size `m` and `n` respectively, return **the median** of the two sorted arrays.
    
    The overall run time complexity should be `O(log (m+n))`.
    
    **Example 1:**
    
    ```
    Input: nums1 = [1,3], nums2 = [2]
    Output: 2.00000
    Explanation: merged array = [1,2,3] and median is 2.
    
    ```
    
    **Example 2:**
    
    ```
    Input: nums1 = [1,2], nums2 = [3,4]
    Output: 2.50000
    Explanation: merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%204.png)
    
    ```python
    class Solution:
        def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
            if len(nums1) <= len(nums2):
                A = nums1
                B = nums2
            else:
                A = nums2
                B = nums1
            # total numbers
            total = len(A) + len(B)
            half = total // 2
            
            # always do the binary search on A
            l, r = 0, len(A) - 1
            while True:
                i = l + (r - l) // 2    # middle idx of A
                j = half - i - 2        # i, j are idxs, start from zero, need to subtract
                
                Aleft = A[i] if i >= 0 else float("-infinity")
                Aright = A[i + 1] if i + 1 < len(A) else float("infinity")
                Bleft = B[j] if j >= 0 else float("-infinity")
                Bright = B[j + 1] if j + 1 < len(B) else float("infinity")
                
                if Aleft <= Bright and Bleft <= Aright:
                    # odd case
                    if total % 2:
                        return min(Aright, Bright)
                    else:
                        return (max(Aleft, Bleft) + min(Aright, Bright)) / 2
                elif Aleft > Bright:
                    # A is too large, need to shrink
                    r = i - 1
                else:
                    l = i + 1
    ```
    

# **Linked List**

## **Easy**

- **[21.Merge two linked lists](https://leetcode.com/problems/merge-two-sorted-lists/)**
    - Method 1
    
    ```python
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    class Solution:
        def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
            # would not worry about the edge cases of inserting to an emty list by creating a dummy node
            dummy = ListNode()
            tail = dummy
    
            while list1 and list2:
                if list1.val <= list2.val:
                    tail.next = list1
                    list1 = list1.next
                else:
                    tail.next = list2
                    list2 = list2.next
                tail = tail.next
    
            if list1:
                tail.next = list1
            elif list2:
                tail.next = list2
    
            return dummy.next
    ```
    
    - Method 2--recursively
    
    ```
    class Solution(object):
        def mergeTwoLists(self, list1, list2):
            """
            :type list1: Optional[ListNode]
            :type list2: Optional[ListNode]
            :rtype: Optional[ListNode]
            """
            # define stop condition
            if not list1: return list2
            if not list2: return list1
            if list1.val <= list2.val:
                list1.next = self.mergeTwoLists(list1.next, list2)
                return list1
            else:
                list2.next = self.mergeTwoLists(list1, list2.next)
                return list2
    ```
    
- **[141. Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)**
    
    Involves the [Floyd’s Tortoise & Hare algorithm](https://www.notion.so/Floyd-s-Tortoise-Hare-algorithm-5988b6a0c2c14d129f3b2c5d8a17e555).
    
    ```python
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.next = None
    
    class Solution:
        def hasCycle(self, head: Optional[ListNode]) -> bool:
            # set a slow and a fast pointer
            slow, fast = head, head
            
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
                if slow == fast:
                    return True
    ```
    

## **Medium**

- **[143. Reorder List](https://leetcode.com/problems/reorder-list/)**
    
    ```python
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    class Solution:
        def reorderList(self, head: Optional[ListNode]) -> None:
            """
            Do not return anything, modify head in-place instead.
            """
            # find middle
            slow, fast = head, head.next
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
    
            # reverse second half
            second = slow.next
            prev = slow.next = None
            while second:
                temp = second.next
                second.next = prev
                prev = second
                second = temp
    
            # merge two halfs
            first, second = head, prev
            while second:
                temp1, temp2 = first.next, second.next
                first.next = second
                second.next = temp1
                first, second = temp1, temp2
    ```
    
- **[19. Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)**
    
    ```python
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    class Solution:
        def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
            dummy = ListNode(0, head)
            left = dummy
            right = head
    
            while n > 0 and right:
                right = right.next
                n -= 1
    
            while right:
                left = left.next
                right = right.next
    
            left.next = left.next.next
            return dummy.next
    
    ```
    
- **[138. Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/)**
    
    ```python
    """
    # Definition for a Node.
    class Node:
        def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
            self.val = int(x)
            self.next = next
            self.random = random
    """
    
    class Solution:
        def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
            # two steps
            # 1. create a hashmap
            oldToCopy = {None: None}
            
            cur = head
            while cur:
                copy = Node(cur.val)
                oldToCopy[cur] = copy
                cur = cur.next
            
            # make the link
            cur = head
            while cur:
                copy = oldToCopy[cur]
                copy.next = oldToCopy[cur.next]
                copy.random = oldToCopy[cur.random]
                cur = cur.next
            
            return oldToCopy[head]
    ```
    
- **[2. Add Two Numbers](https://leetcode.com/problems/add-two-numbers/)**
    
    ```python
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    class Solution:
        def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
            dummy = ListNode()
            cur = dummy
            
            carry = 0
            while l1 or l2:
                v1 = l1.val if l1 else 0
                v2 = l2.val if l2 else 0
                val = v1 + v2 + carry
                carry = val // 10
                val = val % 10
                
                cur.next = ListNode(val)
                cur = cur.next
                l1 = l1.next if l1 else None
                l2 = l2.next if l2 else None
            
            if carry == 1:
                cur.next = ListNode(1)
            
            return dummy.next
    ```
    
- **[287. Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)**
    
    目前没有完全理解，只能牢记解题思路，如下：
    
    1. set a slow and a fast pointer, slow pointer jump 1 step each time, fast pointer jumps 2 steps each time, until the pointers meet each other;
    2. leave the slow pointer there, done with the fast pointer. Put a second pointer at the beginning, keep shifting both slow pointer by 1 until they meet each other.
    3. the second intersection would be the result
    
    没错，就是那么神奇～
    
    ```python
    class Solution:
        def findDuplicate(self, nums: List[int]) -> int:
            # find the first intersection using two pointers
            slow, fast = 0, 0
            while True:
                slow = nums[slow]
                fast = nums[nums[fast]] # 2steps/jump
                if slow == fast:
                    break
            
            slow2 = 0
            while True:
                slow = nums[slow]
                slow2 = nums[slow2]
                while slow == slow2:
                    return slow
    ```
    
- **[146. LRU Cache](https://leetcode.com/problems/lru-cache/)**
    
    高频考题….还需要好好梳理思路（目前跟着答案写出来）
    
    ```python
    class Node:
        def __init__(self, key, val):
            # create a double link list
            # the element of each node includes key and value
            self.key, self.val = key, val
            self.prev = self.next = None  # double link
            
    
    class LRUCache:
    
        def __init__(self, capacity: int):
            self.cap = capacity
            self.cache = {} # map key to node
            
            self.left, self.right = Node(0, 0), Node(0, 0)  # initialize two nodes
            self.left.next, self.right.prev = self.right, self.left
            
    
        def get(self, key: int) -> int:
            if key in self.cache:
                # remove from the list
                self.remove(self.cache[key])
                # insert at right --> becoming a more freqently used key
                self.insert(self.cache[key])
                return self.cache[key].val
        
            # if not exist
            return -1
        
    
        def put(self, key: int, value: int) -> None:
            if key in self.cache:
                # update the value
                self.remove(self.cache[key])
            self.cache[key] = Node(key, value)
            self.insert(self.cache[key])    # insert to the link list
            
            if len(self.cache) > self.cap:
                # remove from the list and delete the LRU from hashmap
                lru = self.left.next    # the left node is the least recently used
                self.remove(lru)
                del self.cache[lru.key]
        
        
        def remove(self, node): # remove node from list
            prev, nxt = node.prev, node.next
            prev.next, nxt.prev = nxt, prev
            
            
        
        def insert(self, node): # insert node at right
            prev, nxt = self.right.prev, self.right
            prev.next = nxt.prev = node
            node.next, node.prev = nxt, prev
            
    
            
            
    # Your LRUCache object will be instantiated and called as such:
    # obj = LRUCache(capacity)
    # param_1 = obj.get(key)
    # obj.put(key,value)
    ```
    

## Hard

- **[23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)**
    
    You are given an array of `k` linked-lists `lists`, each linked-list is sorted in ascending order.
    
    *Merge all the linked-lists into one sorted linked-list and return it.*
    
    **Example 1:**
    
    ```
    Input: lists = [[1,4,5],[1,3,4],[2,6]]
    Output: [1,1,2,3,4,4,5,6]
    Explanation: The linked-lists are:
    [
      1->4->5,
      1->3->4,
      2->6
    ]
    merging them into one sorted list:
    1->1->2->3->4->4->5->6
    
    ```
    
    **Example 2:**
    
    ```
    Input: lists = []
    Output: []
    
    ```
    
    **Example 3:**
    
    ```
    Input: lists = [[]]
    Output: []
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%205.png)
    
    ```python
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    class Solution:
        def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
            if not lists or len(lists) == 0:
                return None
            
            # merge two list each time  
            while len(lists) > 1:
                mergeLists = []
                for i in range(0, len(lists), 2):
                    l1 = lists[i]
                    l2 = lists[i + 1] if (i + 1) < len(lists) else None
                    mergeLists.append(self.mergeList(l1, l2))
                lists = mergeLists    
            return lists[0]
        
        def mergeList(self, l1, l2):
            dummy = ListNode(0)
            head = dummy
            
            while l1 and l2:
                if l1.val <= l2.val:
                    head.next = l1
                    l1 = l1.next if l1 else None
                else:
                    head.next = l2
                    l2 = l2.next if l2 else None
                head = head.next
            
            if l1:
                head.next = l1
            elif l2:
                head.next = l2
            
            return dummy.next
    ```
    
- **[25. Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/)**
    
    Given the `head` of a linked list, reverse the nodes of the list `k` at a time, and return *the modified list*.
    
    `k` is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of `k` then left-out nodes, in the end, should remain as it is.
    
    You may not alter the values in the list's nodes, only nodes themselves may be changed.
    
    **Example 1:**
    
    ![https://assets.leetcode.com/uploads/2020/10/03/reverse_ex1.jpg](https://assets.leetcode.com/uploads/2020/10/03/reverse_ex1.jpg)
    
    ```
    Input: head = [1,2,3,4,5], k = 2
    Output: [2,1,4,3,5]
    
    ```
    
    **Example 2:**
    
    ![https://assets.leetcode.com/uploads/2020/10/03/reverse_ex2.jpg](https://assets.leetcode.com/uploads/2020/10/03/reverse_ex2.jpg)
    
    ```
    Input: head = [1,2,3,4,5], k = 3
    Output: [3,2,1,4,5]
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%206.png)
    
    ```python
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    class Solution:
        def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
            dummy = ListNode(0, head)
            groupPrev = dummy # store the previous node of each group
            
            while True:
                kth = self.getKth(groupPrev, k)
                if not kth: # if not enough k nodes 
                    break
                groupNext = kth.next    # store the node after the original kth node
                
                # in-group reverse
                prev, cur = kth.next, groupPrev.next
                while cur != groupNext:
                    temp = cur.next # point the first node to the first node in next group
                    cur.next = prev
                    prev = cur
                    cur = temp
                
                # out-group reverse
    						# rightnow, the kth node reversed to the front
    						# should point the groupPrev node.next to the kth node
                temp = groupPrev.next
                groupPrev.next = kth
                groupPrev = temp
            
            return dummy.next
        
        def getKth(self, cur, k):
            # get the last node of a group
            while cur and k > 0:
                cur = cur.next
                k -= 1
            return cur
    ```
    

# **Trees**

## **Easy**

- **[226.](notion://www.notion.so/leahishere/226/Invert%20Binary%20Tree)[Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)**
    - everytime visit a node, look at its two children and swap their position
    - recursively run invert on left and right trees
    - DFS search--depth first search
    
    ```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
            if not root:
                return None
    
            # swap the children
            tmp = root.left
            root.left = root.right
            root.right = tmp
    
            self.invertTree(root.left)
            self.invertTree(root.right)
    
            return root
    ```
    
- **[104.Maximum Depth of Binary Tree](notion://www.notion.so/leahishere/104/Maximum%20Depth%20of%20Binary%20Tree)**
    
    Method-1: iteratively DFS
    
    - depth-first-search without using recursion
    - using a stack structure
    - doing by preorder: adding the left and right subtree
    
    ```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def maxDepth(self, root: Optional[TreeNode]) -> int:
            stack = [[root, 1]]
            res = 0
    
            while stack:
                node, depth = stack.pop()
    
                if node:
                    res = max(res, depth)
                    stack.append([node.left, depth+1])
                    stack.append([node.right, depth+1])
    
            return res
    ```
    
    Method-2: recursively DFS （the best solution)
    
    - the simplest way
    - base case: empty tree --> return 0
    - 1 + max(dfs(left), dfs(right))
        
        t: O(n) search through the tree
        
        m: O(n)
        
    
    ```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def maxDepth(self, root: Optional[TreeNode]) -> int:
            if not root:
                return 0
    
            return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
    ```
    
    Method-3: iteratively BFS(Breadth-first-search)
    
    - couting the number of levels we have
    - the number of level == maxmum depth of tree
    - put the root in to Queue
    
    ```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def maxDepth(self, root: Optional[TreeNode]) -> int:
            if not root:	# the base case
                return 0
    
            level = 0
            q = deque([root])   # initialize the Queue with the single value
            while q:	# keep going untill the queue is empty
    
                # remove all element in the queue and add their childen
                for i in range(len(q)):
                    node = q.popleft()  # remove from the left
                    if node.left:
                        q.append(node.left)
                    if node.right:
                        q.append(node.right)
    
                level += 1
            return level
    ```
    
- **[543.Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/)**
    - search from the bottom nodes
    
    ```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
            res = [0]   # a global result variable
    
            def dfs(root):
                # return the height
                if not root:
                    return -1   # for a no tree; 0 for a single node
                left = dfs(root.left)
                right = dfs(root.right)
    
                res[0] = max(res[0], 2 + left + right)
    
                return 1 + max(left, right)
    
            dfs(root)
            return res[0]
    ```
    
    time: O(n)
    
- **[110.Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/)**
    
    ```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def isBalanced(self, root: Optional[TreeNode]) -> bool:
    
            def dfs(root):
                if not root:
                    return [True, 0]
    
                left, right = dfs(root.left), dfs(root.right)
                balanced = (left[0] and right[0] and
                           abs(left[1]-right[1])<=1)
    
                return [balanced, 1+max(left[1], right[1])]
    
            return dfs(root)[0]
    ```
    
- **[100.SameTree](https://leetcode.com/problems/same-tree/)**
    
    ```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
            if not p and not q:
                return True
            if not p or not q or p.val != q.val:
                return False
    
            return (self.isSameTree(p.left, q.left) and
                    self.isSameTree(p.right, q.right))
    ```
    
- **[572.Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/)**
    
    ```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def isSubtree(self, s: Optional[TreeNode], t: Optional[TreeNode]) -> bool:
            # the order condition is very important
            if not t: return True
            if not s: return False
    
            if self.sameTree(s, t):
                return True
    
            else:
                return (self.isSubtree(s.left, t) or
                        self.isSubtree(s.right, t))
    
        def sameTree(self, s, t):
            if not s and not t:
                return True
    
            if s and t and s.val == t.val:
                return (self.sameTree(s.left, t.left) and
                        self.sameTree(s.right, t.right))
    
            return False
    ```
    
- **[235.](notion://www.notion.so/leahishere/235/Lowest%20Common%20Ancestor%20of%20a%20Binary%20Search%20Tree)[Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)**
    - do not neet to search for the entire tree
        - search for only a one node in each level
        - time log(n)
        - space O(1)
    
    ```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.left = None
    #         self.right = None
    
    class Solution:
        def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
            cur = root
    
            while cur:
                if p.val > cur.val and q.val > cur.val:
                    cur = cur.right
                elif p.val < cur.val and q.val < cur.val:
                    cur = cur.left
                else:
                    return cur
    ```
    

## **Medium**

- **[102. Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)**
    
    t: O(n)
    
    m: O(n)
    
    ```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
            res = []
            q = collections.deque() # initialize a queue
            q.append(root)
            
            while q:
                qLen = len(q)
                level = []  # result of each level
                for i in range(qLen):
                    node = q.popleft()
                    if node:
                        level.append(node.val)
                        q.append(node.left)
                        q.append(node.right)
                if level:
                    res.append(level)
            
            return res
    ```
    
- **[199. Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view/)**
    
    ```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
            res = []
            q = collections.deque()
            q.append(root)
            
            while q:
                rightSide = None
                qLen = len(q)
                for i in range(qLen):
                    node = q.popleft()
                    if node:
                        rightSide = node
                        q.append(node.left)
                        q.append(node.right)
                
                if rightSide:
                    res.append(rightSide.val)
                    
            return res
    ```
    
- **[1448. Count Good Nodes in Binary Tree](https://leetcode.com/problems/count-good-nodes-in-binary-tree/) —> Microsoft most frequent problem in 2021**
    - solve recursively
    
    ```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def goodNodes(self, root: TreeNode) -> int:
            
            def dfs(node, maxVal):
                if not node:
                    return 0
                
                res = 1 if node.val >= maxVal else 0
                maxVal = max(node.val, maxVal)
                
                res += dfs(node.left, maxVal)
                res += dfs(node.right, maxVal)
                
                return res
            
            return dfs(root, root.val)
    ```
    
- **[98. Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)**
    
    ```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def isValidBST(self, root: Optional[TreeNode]) -> bool:
            
            def dfs(node, left, right):
                # pass the left and right boundary in
                if not node:
                    return True # a Null tree is also a valid binary tree
                
                # the node value should less than the left boundary & greater than the right boundary
                if not (node.val < right and node.val > left):  
                    return False
                
                return (dfs(node.left, left, node.val) and  # update right boundary for left subtree
                        dfs(node.right, node.val, right))   # update left boundary for right subtree
            
            return dfs(root, float("-inf"), float("inf"))   # the boundary for root node from negative infinity to positive inf
    ```
    
- **[230. Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)**
    
    ```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
            # visit the tree in order
            stack = []
            n = 0
            cur = root
            
            while cur or stack:
                while cur:
                    stack.append(cur)
                    cur = cur.left
                
                cur = stack.pop()
                n += 1
                if n == k:
                    return cur.val
                cur = cur.right
    ```
    
- **[105. Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)**
    
    ```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
            if not preorder or not inorder:
                return None
            
            # recursively
            root = TreeNode(preorder[0])
            mid = inorder.index(preorder[0])
            
            root.left = self.buildTree(preorder[1: mid+1], inorder[:mid])
            root.right = self.buildTree(preorder[mid+1: ], inorder[mid+1:])
            
            return root
    ```
    

## Hard

- **[124. Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)**
    
    A **path** in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence **at most once**. Note that the path does not need to pass through the root.
    
    The **path sum** of a path is the sum of the node's values in the path.
    
    Given the `root` of a binary tree, return *the maximum **path sum** of any **non-empty** path*.
    
    **Example 1:**
    
    ![https://assets.leetcode.com/uploads/2020/10/13/exx1.jpg](https://assets.leetcode.com/uploads/2020/10/13/exx1.jpg)
    
    ```
    Input: root = [1,2,3]
    Output: 6
    Explanation: The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.
    
    ```
    
    **Example 2:**
    
    ![https://assets.leetcode.com/uploads/2020/10/13/exx2.jpg](https://assets.leetcode.com/uploads/2020/10/13/exx2.jpg)
    
    ```
    Input: root = [-10,9,20,null,null,15,7]
    Output: 42
    Explanation: The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 = 42.
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%207.png)
    
    ```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def maxPathSum(self, root: Optional[TreeNode]) -> int:
            res = [root.val] # global variable
            
            # return max path sum without split
            def dfs(root):
                if not root:
                    return 0
                
                maxLeft = dfs(root.left)
                maxRight = dfs(root.right)
                
                maxLeft = max(maxLeft, 0)
                maxRight = max(maxRight, 0)
                
                # with split, update res directly
                res[0] = max(res[0], root.val + maxLeft + maxRight)
                
                return root.val + max(maxLeft, maxRight)
            
            dfs(root)
            return res[0]
    ```
    
- **[297. Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)**
    
    Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.
    
    Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.
    
    **Clarification:** The input/output format is the same as [how LeetCode serializes a binary tree](https://support.leetcode.com/hc/en-us/articles/360011883654-What-does-1-null-2-3-mean-in-binary-tree-representation-). You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.
    
    **Example 1:**
    
    ![https://assets.leetcode.com/uploads/2020/09/15/serdeser.jpg](https://assets.leetcode.com/uploads/2020/09/15/serdeser.jpg)
    
    ```
    Input: root = [1,2,3,null,null,4,5]
    Output: [1,2,3,null,null,4,5]
    
    ```
    
    **Example 2:**
    
    ```
    Input: root = []
    Output: []
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%208.png)
    
    ```python
    # Definition for a binary tree node.
    # class TreeNode(object):
    #     def __init__(self, x):
    #         self.val = x
    #         self.left = None
    #         self.right = None
    
    class Codec:
    
        def serialize(self, root):
            """Encodes a tree to a single string.
            
            :type root: TreeNode
            :rtype: str
            """
            res = []  # store the strings
            def dfs(node):
                if not node:
                    res.append('N')
                    return
                res.append(str(node.val))
                dfs(node.left)
                dfs(node.right)
            dfs(root)
            return ','.join(res)
        
        def deserialize(self, data):
            """Decodes your encoded data to tree.
            
            :type data: str
            :rtype: TreeNode
            """
            vals = data.split(',')
            self.i = 0 # create a pointer
            
            def dfs():
                if vals[self.i] == 'N':
                    # reach the end of left or right subtree
                    self.i += 1
                    return None
                
                node = TreeNode(int(vals[self.i]))
                self.i += 1
                node.left = dfs()
                node.right = dfs()
                return node
            return dfs()
                
            
    
    # Your Codec object will be instantiated and called as such:
    # ser = Codec()
    # deser = Codec()
    # ans = deser.deserialize(ser.serialize(root))
    ```
    

# Tries

## Medium

- **[208. Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)**
    
    t: O(26) —> O(1)
    
    ```python
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.endOfWord = False
    
    class Trie:
    
        def __init__(self):
            self.root = TrieNode()
            
    
        def insert(self, word: str) -> None:
            cur = self.root
            
            for c in word:
                if c not in cur.children:
                    cur.children[c] = TrieNode()
                cur = cur.children[c]
            cur.endOfWord = True
            
    
        def search(self, word: str) -> bool:
            cur = self.root
            
            for c in word:
                if c not in cur.children:
                    return False
                cur = cur.children[c]
            
            return cur.endOfWord
        
    
        def startsWith(self, prefix: str) -> bool:
            cur = self.root
            
            for c in prefix:
                if c not in cur.children:
                    return False
                cur = cur.children[c]
            
            return True
    
    # Your Trie object will be instantiated and called as such:
    # obj = Trie()
    # obj.insert(word)
    # param_2 = obj.search(word)
    # param_3 = obj.startsWith(prefix)
    ```
    
- **[211. Design Add and Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure/)**
    
    用到了递归。
    
    ```python
    class TrieNode:
        
        def __init__(self):
            self.children = {}
            self.endOfWord = False
        
    class WordDictionary:
    
        def __init__(self):
            self.root = TrieNode()
    
        def addWord(self, word: str) -> None:
            cur = self.root
            
            for c in word:
                if c not in cur.children:
                    cur.children[c] = TrieNode()
                cur = cur.children[c]
            
            cur.endOfWord = True
    
        def search(self, word: str) -> bool:
            
            def dfs(j, root):
                cur = root
                
                for i in range(j, len(word)):
                    c = word[i]
                    
                    if c == ".":
                        for child in cur.children.values():
                            if dfs(i+1, child):
                                return True
                        return False
                        
                    else:
                        if c not in cur.children:
                            return False
                        cur = cur.children[c]
                return cur.endOfWord
            
            return dfs(0, self.root)
    
    # Your WordDictionary object will be instantiated and called as such:
    # obj = WordDictionary()
    # obj.addWord(word)
    # param_2 = obj.search(word)
    ```
    

## Hard

- **[212. Word Search II](https://leetcode.com/problems/word-search-ii/)**
    
    Given an `m x n` `board` of characters and a list of strings `words`, return *all words on the board*.
    
    Each word must be constructed from letters of sequentially adjacent cells, where **adjacent cells** are horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.
    
    **Example 1:**
    
    ![https://assets.leetcode.com/uploads/2020/11/07/search1.jpg](https://assets.leetcode.com/uploads/2020/11/07/search1.jpg)
    
    ```
    Input: board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]
    Output: ["eat","oath"]
    
    ```
    
    **Example 2:**
    
    ![https://assets.leetcode.com/uploads/2020/11/07/search2.jpg](https://assets.leetcode.com/uploads/2020/11/07/search2.jpg)
    
    ```
    Input: board = [["a","b"],["c","d"]], words = ["abcb"]
    Output: []
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%209.png)
    
    ```python
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.isWord = False
            self.refs = 0
        
        def addWord(self, word):
            cur = self
            self.refs += 1
            for w in word:
                if w not in cur.children:
                    cur.children[w] = TrieNode()
                cur = cur.children[w]
                cur.refs += 1
            cur.isWord = True
        
        def removeWord(self, word):
            cur = self
            cur.refs -= 1
            for w in word:
                if w in cur.children:
                    cur = cur.children[w]
                    cur.refs -= 1
    
    class Solution:
        def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
            root = TrieNode()
            for w in words:
                root.addWord(w)
            
            ROWS, COLS = len(board), len(board[0])
            res, visit = set(), set()
    
            def dfs(r, c, node, word):
                if (
                    r not in range(ROWS) or
                    c not in range(COLS) or
                    board[r][c] not in node.children or
                    node.children[board[r][c]].refs < 1 or
                    (r, c) in visit
                ):
                    return
                
                visit.add((r, c))
                node = node.children[board[r][c]]
                word += board[r][c]
                if node.isWord:
                    node.isWord = False
                    res.add(word)
                    root.removeWord(word)
                
                dfs(r + 1, c, node, word)
                dfs(r, c + 1, node, word)
                dfs(r - 1, c, node, word)
                dfs(r, c - 1, node, word)
                visit.remove((r, c))
            
            for r in range(ROWS):
                for c in range(COLS):
                    dfs(r, c, root, "")
            
            return list(res)
    ```
    

# **Heap/ Priority Queue**

## **easy**

- **[7](notion://www.notion.so/leahishere/703/Kth%20Largest%20Element%20in%20a%20Stream)[03.K-th Largest Element in a Stream](https://leetcode.com/problems/kth-largest-element-in-a-stream/)**
    - stream: could add number to a list of numbers continuously
    - MinHeap data structure -- min heap of size K
        - has sorted property
        - could add and pop element in time: o(log(n))
        - could get the min value in time o(1)
    
    ```python
    class KthLargest:
    
        def __init__(self, k: int, nums: List[int]):
            # minHeap with K largest integers
            self.minHeap, self.k = nums, k  # right now is just an array
            heapq.heapify(self.minHeap)
            while len(self.minHeap) > k:
                heapq.heappop(self.minHeap)
    
        def add(self, val: int) -> int:
            heapq.heappush(self.minHeap, val)
            if len(self.minHeap) > self.k:
                heapq.heappop(self.minHeap)
    
            return self.minHeap[0]
    ```
    
- **[104](notion://www.notion.so/leahishere/1046/Last%20Stone%20Weight)[6.Last Stone Weight](https://leetcode.com/problems/last-stone-weight/)**
    - use a MaxHeap
        - do not exist in python, using minHeap to simulate MaxHeap by multiplying -1
        - time to construct maxHeap O(n)
        - time to get the max value O(logn)
    - overall time: o(nlogn)
    
    ```python
    class Solution:
        def lastStoneWeight(self, stones: List[int]) -> int:
            stones = [-s for s in stones]
            heapq.heapify(stones)
    
            while len(stones) >= 2:
                w1 = -heapq.heappop(stones)
                w2 = -heapq.heappop(stones)
    
                if w1 != w2:
                    w = w1 - w2
                    heapq.heappush(stones, -w)
    
            if len(stones) == 1:
                return -stones[0]
            else:
                return 0
    ```
    

## Medium

- **[973. K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/)**
    
    ```python
    class Solution:
        def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
            pts = []
            res = []
            heapq.heapify(pts)
            
            for p in points:
                d = math.sqrt(p[0]**2 + p[1]**2)
                heapq.heappush(pts, [d, p])
            
            for i in range(k):
                p = heapq.heappop(pts)[1]
                res.append(p)
            return res
    ```
    
- **[215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)**
    - Solution 1: maxHeap t:O(n+klog(n))
    
    ```python
    class Solution:
        def findKthLargest(self, nums: List[int], k: int) -> int:
            heap = [-n for n in nums]
            heapq.heapify(heap)
            for i in range(k):
                num = heapq.heappop(heap)
            
            return -num
    ```
    
    - Solution 2: QuickSelect t: O(n)
    
    ```python
    class Solution:
        def findKthLargest(self, nums: List[int], k: int) -> int:
            res_idx = len(nums) - k
            
            def quickSelect(l, r):
                if l == r:
                    return nums[l]
                
                pivot = nums[r]
                p = l
                
                for i in range(l, r):
                    if nums[i] <= pivot:
                        nums[p], nums[i] = nums[i], nums[p]
                        p += 1
                    
                nums[p], nums[r] = nums[r], nums[p]
                
                if p > res_idx: 
                    return quickSelect(l, p - 1)
                elif p < res_idx:
                    return quickSelect(p + 1, r)
                else:
                    return nums[p]
            
            return quickSelect(0, len(nums)-1)
    ```
    
- **[621. Task Scheduler](https://leetcode.com/problems/task-scheduler/)**
    
    ```python
    class Solution:
        def leastInterval(self, tasks: List[str], n: int) -> int:
            count = Counter(tasks)  # count total number of each task, a dictionary
            maxHeap = [-cnt for cnt in count.values()]
            heapq.heapify(maxHeap)
            
            time = 0
            q = deque() # pairs of [-cnt, idleTime]
            
            while q or maxHeap:
                time += 1
                if maxHeap:
                    cnt = 1 + heapq.heappop(maxHeap)
                    if cnt:
                        q.append([cnt, time+n])
                    
                if q and q[0][1] == time:
                    heapq.heappush(maxHeap, q.popleft()[0])
            
            return time
    ```
    
- **[355. Design Twitter](https://leetcode.com/problems/design-twitter/)**
    
    ```python
    class Twitter:
    
        def __init__(self):
            self.count = 0
            self.tweetMap = defaultdict(list)   # userId -> list of [count, tweetIds]
            self.followMap = defaultdict(set)   # userId -> set of followeeId
    
        def postTweet(self, userId: int, tweetId: int) -> None:
            self.tweetMap[userId].append([self.count, tweetId])
            self.count -= 1
    
        def getNewsFeed(self, userId: int) -> List[int]:
            res = []
            minHeap = []
            
            # add the user-self to the followMap
            self.followMap[userId].add(userId)
            
            # add every followee's last post to the minHeap
            for followeeId in self.followMap[userId]:
                if followeeId in self.tweetMap:   **# Do NOT forget to check if the followeeId has posted anything before**
                    index = len(self.tweetMap[followeeId]) - 1
                    count, tweetId = self.tweetMap[followeeId][index]
                    heapq.heappush(minHeap, [count, tweetId, followeeId, index - 1])
            
            while minHeap and len(res) < 10:
                count, tweetId, followeeId, index = heapq.heappop(minHeap)  # index is the next position to add
                res.append(tweetId)
                if index >= 0:
                    count, tweetId = self.tweetMap[followeeId][index]
                    heapq.heappush(minHeap, [count, tweetId, followeeId, index - 1])
                
            return res
                
            
        def follow(self, followerId: int, followeeId: int) -> None:
            self.followMap[followerId].add(followeeId)
    
        def unfollow(self, followerId: int, followeeId: int) -> None:
            # check if the followee has been followed
            if followeeId in self.followMap[followerId]:
                self.followMap[followerId].remove(followeeId)
    
    # Your Twitter object will be instantiated and called as such:
    # obj = Twitter()
    # obj.postTweet(userId,tweetId)
    # param_2 = obj.getNewsFeed(userId)
    # obj.follow(followerId,followeeId)
    # obj.unfollow(followerId,followeeId)
    ```
    

## Hard

- **[295. Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)**
    - Problem Description
        
        The **median** is the middle value in an ordered integer list. If the size of the list is even, there is no middle value and the median is the mean of the two middle values.
        
        - For example, for `arr = [2,3,4]`, the median is `3`.
        - For example, for `arr = [2,3]`, the median is `(2 + 3) / 2 = 2.5`.
        
        Implement the MedianFinder class:
        
        - `MedianFinder()` initializes the `MedianFinder` object.
        - `void addNum(int num)` adds the integer `num` from the data stream to the data structure.
        - `double findMedian()` returns the median of all elements so far. Answers within `105` of the actual answer will be accepted.
    - Examples
        
        **Example 1:**
        
        ```
        Input
        ["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
        [[], [1], [2], [], [3], []]
        Output
        [null, null, null, 1.5, null, 2.0]
        
        Explanation
        MedianFinder medianFinder = new MedianFinder();
        medianFinder.addNum(1);    // arr = [1]
        medianFinder.addNum(2);    // arr = [1, 2]
        medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
        medianFinder.addNum(3);    // arr[1, 2, 3]
        medianFinder.findMedian(); // return 2.0
        ```
        
    - Solution
        
        ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2010.png)
        
    - Code
        
        ```python
        class MedianFinder:
        
            def __init__(self):
                # two heaps
                # small heap is a maxHeap
                # large heap is a minHeap
                self.small, self.large = [], []
        
            def addNum(self, num: int) -> None:
                # add to the small heap by default
                heapq.heappush(self.small, num * -1)
                
                # check the max of small <= min of large
                if self.small and self.large and (-1 * self.small[0]) > self.large[0]:
                    # if not, add the large in small to large
                    val = -1 * heapq.heappop(self.small)
                    heapq.heappush(self.large, val)
                
                # check the length
                if len(self.small) - len(self.large) > 1:
                    # small heap has more nums
                    val = -1 * heapq.heappop(self.small)
                    heapq.heappush(self.large, val)
                if len(self.large) - len(self.small) > 1:
                    val = heapq.heappop(self.large)
                    heapq.heappush(self.small, -1 * val)
        
            def findMedian(self) -> float:
                if len(self.small) == len(self.large):
                    return ((-1 * self.small[0]) + self.large[0])/2
                elif len(self.small) > len(self.large):
                    return -1 * self.small[0]
                else:
                    return self.large[0]
        
        # Your MedianFinder object will be instantiated and called as such:
        # obj = MedianFinder()
        # obj.addNum(num)
        # param_2 = obj.findMedian()
        ```
        

# [Backtracking](https://www.notion.so/BackingTracking-01d5f74580b748fa8647794e97bd39bb)

## Medium

- **[78. Subsets](https://leetcode.com/problems/subsets/)**
    
    ```python
    class Solution:
        def subsets(self, nums: List[int]) -> List[List[int]]:
            # two options for each num: 
            # 1/ add the number into subset
            # 2/ not adding the number into subset
            res = []
            subset = []
            
            def dfs(i):
                # pass the index into the function
                if i >= len(nums):
                    # out of bound
                    res.append(subset.copy())
                    return 
                
                # decision to include nums[i]
                subset.append(nums[i])
                dfs(i + 1)
                
                # decision NOT to include nums[i]
                subset.pop()    # remove the number we just added
                dfs(i + 1)
            
            dfs(0)
            return res
    ```
    
- **[39. Combination Sum](https://leetcode.com/problems/combination-sum/)**
    
    ```python
    class Solution:
        def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
            res = []
            
            def dfs(i, cur, total):
                if total == target:
                    res.append(cur.copy())
                    return 
                if i >= len(candidates) or total > target:
                    return
                
                cur.append(candidates[i])
                dfs(i, cur, total + candidates[i])
                cur.pop()
                dfs(i + 1, cur, total)
                
            dfs(0, [], 0)
            return res
    ```
    
- **[46. Permutations](https://leetcode.com/problems/permutations/)**
    
    ```python
    class Solution:
        def permute(self, nums: List[int]) -> List[List[int]]:
            res = []
            
            # base case
            if len(nums) == 1:
                return [nums[:]]    # deepcopy
            
            for i in range(len(nums)):
                n = nums.pop(0) # pop from index 0
                perms = self.permute(nums)
                
                for perm in perms:
                    perm.append(n)
                res.extend(perms)
                nums.append(n)
            
            return res
    ```
    
- **[90. Subsets II](https://leetcode.com/problems/subsets-ii/)**
    
    ```python
    class Solution:
        def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
            res = []
            nums.sort()
            
            def dfs(i, cur):
                if i >= len(nums):
                    res.append(cur.copy())
                    return
                
                # contain nums[i]
                cur.append(nums[i])
                dfs(i + 1, cur)
                
                # NOT contain nums[i]
                cur.pop()
                while i + 1 < len(nums) and nums[i] == nums[i + 1]:
                    i += 1
                dfs(i + 1, cur)
                
            dfs(0, [])
            return res
    ```
    
- **[40. Combination Sum II](https://leetcode.com/problems/combination-sum-ii/)**
    
    ```python
    class Solution:
        def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
            candidates.sort()
            
            res = []
            def backtrack(cur, pos, target):
                if target == 0:
                    res.append(cur.copy())
                if target <= 0:
                    return
                
                prev = -1
                for i in range(pos, len(candidates)):
                    if candidates[i] == prev:
                        continue
                    cur.append(candidates[i])
                    backtrack(cur, i + 1, target - candidates[i])
                    cur.pop()
                    prev = candidates[i]
                    
            backtrack([], 0, target)
            return res
    ```
    
- **[79. Word Search](https://leetcode.com/problems/word-search/)**
    
    ```python
    class Solution:
        def exist(self, board: List[List[str]], word: str) -> bool:
            ROWS, COLS = len(board), len(board[0])
            path = set() # 存储已经访问过的路径
            
            def dfs(r, c, i):
                if i == len(word):
                    return True
                if (i > len(word) or min(r, c) < 0 or     # 注意边界条件要写全，同时边界条件的顺序也很重要。
                    r >= ROWS or c >= COLS or
                   (r, c) in path or
                    board[r][c] != word[i]):
                    return False
                
                path.add((r, c)) # 将该位置加入路径
                res = (dfs(r + 1, c, i + 1) or
                        dfs(r - 1, c, i + 1) or 
                        dfs(r, c + 1, i + 1) or 
                        dfs(r, c - 1, i + 1))
                path.remove((r, c))
                return res
            
            for r in range(ROWS):
                for c in range(COLS):
                    if dfs(r, c, 0): 
                        return True
            return False
    ```
    
- **[131. Palindrome Partitioning](https://leetcode.com/problems/palindrome-partitioning/)**
    
    ```python
    class Solution:
        def partition(self, s: str) -> List[List[str]]:
            res = []
            cur = []
            
            def dfs(i):
                if i >= len(s):
                    res.append(cur.copy())
                    return
                
                for j in range(i, len(s)):
                    if self.Pali(s, i, j):
                        cur.append(s[i: j + 1])
                        dfs(j + 1)
                        cur.pop()
                        
            dfs(0)
            return res
                
        def Pali(self, s, l, r):
            while l < r:
                if s[l] != s[r]:
                    return False
                l += 1
                r -= 1
            return True
    ```
    
- **[17. Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)**
    
    ```python
    class Solution:
        def letterCombinations(self, digits: str) -> List[str]:
            res = []
            # create a hashmap
            digitToStr = {'2': 'abc',
                          '3': 'def',
                          '4': 'ghi',
                          '5': 'jkl',
                          '6': 'mno',
                          '7': 'pqrs',
                          '8': 'tuv',
                          '9': 'wxyz'}
            
            def dfs(i, cur):
                if len(cur) == len(digits):
                    res.append(cur)
                    return
                
                for c in digitToStr[digits[i]]:
                    dfs(i + 1, cur + c)
                
            if digits:
                dfs(0, "")
            return res
    ```
    

## Hard

- **[51. N-Queens](https://leetcode.com/problems/n-queens/)**
    - Problem
        
        The **n-queens** puzzle is the problem of placing `n` queens on an `n x n` chessboard such that no two queens attack each other.
        
        Given an integer `n`, return *all distinct solutions to the **n-queens puzzle***. You may return the answer in **any order**.
        
        Each solution contains a distinct board configuration of the n-queens' placement, where `'Q'` and `'.'` both indicate a queen and an empty space, respectively.
        
    - Example
        
        **Example 1:**
        
        ![https://assets.leetcode.com/uploads/2020/11/13/queens.jpg](https://assets.leetcode.com/uploads/2020/11/13/queens.jpg)
        
        ```
        Input: n = 4
        Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
        Explanation: There exist two distinct solutions to the 4-queens puzzle as shown above
        
        ```
        
        **Example 2:**
        
        ```
        Input: n = 1
        Output: [["Q"]]
        ```
        
    - Solution
        
        ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2011.png)
        
    - Code
        
        ```python
        class Solution:
            def solveNQueens(self, n: int) -> List[List[str]]:
                col = set()
                posDiag = set()
                negDiag = set()
                
                res = []
                board = [["."] * n for i in range(n)]
                
                def backtrack(r):
                    if r == n:
                        # reach the end of the board
                        copy = ["".join(row) for row in board]
                        res.append(copy)
                        return
                    
                    for c in range(n):
                        if (c in col or
                            (r + c) in posDiag or
                            (r - c) in negDiag):
                            continue
                        
                        col.add(c)
                        posDiag.add((r + c))
                        negDiag.add((r - c))
                        board[r][c] = "Q"
                        
                        backtrack(r + 1)
                        
                        col.remove(c)
                        posDiag.remove((r + c))
                        negDiag.remove((r - c))
                        board[r][c] = '.'
                    
                backtrack(0)
                return res
        ```
        

# Graph

## Medium

- **[200. Number of Islands](https://leetcode.com/problems/number-of-islands/) —> frequent asked problem**
    
    ```python
    class Solution:
        def numIslands(self, grid: List[List[str]]) -> int:
            # use a **breadth-first search**
            # data structure: queue
            if not grid:
                # if no given input, of course no island
                return 0
            
            islands = 0
            visit = set()   # store visited positions 
            rows, cols = len(grid), len(grid[0])
           
            def bfs(r, c):
                visit.add((r, c))
                q = collections.deque()               q.append((r, c))
                directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
                
                while q:
                    row, col = q.popleft()   ##### if change to q.pop(), would become a DFS solution
                    for dr, dc in directions:
                        r, c = row + dr, col + dc
                        if (r in range(rows) and c in range(cols) and
                            grid[r][c] == "1" and (r, c) not in visit):
                            visit.add((r, c))
                            q.append((r, c))
    
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == "1" and (r, c) not in visit:
                        bfs(r, c)
                        islands += 1
            
            return islands
    ```
    
- [**133. Clone Graph**](https://leetcode.com/problems/clone-graph/)
    
    ```python
    # recursively adding neighbors to the oldToNew hashmap
    """
    # Definition for a Node.
    class Node:
        def __init__(self, val = 0, neighbors = None):
            self.val = val
            self.neighbors = neighbors if neighbors is not None else []
    """
    
    class Solution:
        def cloneGraph(self, node: 'Node') -> 'Node':
            # create a hashmap to map the old nodes to new nodes
            oldToNew = {}
            
            def dfs(node):
                if node in oldToNew:
                    return oldToNew[node]
                
                copy = Node(node.val)
                oldToNew[node] = copy
                
                for neighbor in node.neighbors:
                    copy.neighbors.append(dfs(neighbor))
                
                return copy
            
            return dfs(node) if node else None
    ```
    
- **[695. Max Area of Island](https://leetcode.com/problems/max-area-of-island/)**
    
    ```python
    class Solution:
        def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
            if not grid:
                return 0
            
            # similar to the question "number of island"
            maxA = 0
            visit = set()
            rows, cols = len(grid), len(grid[0])
            
            # use breadth-first-search
            def bfs(r, c):
                curA = 1
                visit.add((r, c))
                q = collections.deque()
                q.append((r, c))
                directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
                
                while q:
                    row, col = q.popleft()
                    for dr, dc in directions:
                        r = row + dr
                        c = col + dc
                        if (r in range(rows) and c in range(cols) and
                            grid[r][c] == 1 and (r, c) not in visit):
                            visit.add((r, c))
                            q.append((r, c))
                            curA += 1
                return curA
            
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == 1 and (r, c) not in visit:
                        curA = bfs(r, c)
                        maxA = max(maxA, curA)
            
            return maxA
    ```
    
- **[417. Pacific Atlantic Water Flow](https://leetcode.com/problems/pacific-atlantic-water-flow/)**
    
    ```python
    class Solution:
        def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
            # find grid flows to Pacific and Atlantic Separately using two sets
            # cells in both sets would be the final answer
            
            # go from ocean to land since we know the edge could always flows to the ocean
            # the condition becomes the reverse flow (from lower to higher)
            
            # two sets
            pac, atl = set(), set()
            rows, cols = len(heights), len(heights[0])
            
            def dfs(r, c, visit, PrevHeights):
                if (r not in range(rows) or c not in range(cols) or 
                    (r, c) in visit or 
                    PrevHeights > heights[r][c]):
                    return 
                
                visit.add((r, c))
                dfs(r + 1, c, visit, heights[r][c])
                dfs(r - 1, c, visit, heights[r][c])
                dfs(r, c + 1, visit, heights[r][c])
                dfs(r, c - 1, visit, heights[r][c])
            
            for c in range(cols):
                dfs(0, c, pac, heights[0][c])
                dfs(rows - 1, c, atl, heights[rows - 1][c])
            
            for r in range(rows):
                dfs(r, 0, pac, heights[r][0])
                dfs(r, cols - 1, atl, heights[r][cols - 1])
            
            res = []
            for r in range(rows):
                for c in range(cols):
                    if (r, c) in pac and (r, c) in atl:
                        res.append([r, c])
            return res
    ```
    
- **[130. Surrounded Regions](https://leetcode.com/problems/surrounded-regions/)**
    
    ```python
    class Solution:
        def solve(self, board: List[List[str]]) -> None:
            """
            Do not return anything, modify board in-place instead.
            """
            rows, cols = len(board), len(board[0])
            
            def capture(r, c):
                if (r not in range(rows) or 
                    c not in range(cols) or
                    board[r][c] != "O"):
                    return
            
                board[r][c] = "T"
                capture(r + 1, c)
                capture(r - 1, c)
                capture(r, c + 1)
                capture(r, c - 1)
                    
            
            # 1. capture all unsurrounding regions -> "O" -- "T"
            # turn unsurrounding "O" to a temporal variable "T"
            for r in range(rows):
                for c in range(cols):
                    if ((r in [0, rows - 1] or c in [0, cols - 1]) and
                        board[r][c] == "O"):
                        capture(r, c)
            
            
            # 2. Capture all surrounding regions -> "O" -- "X"
            for r in range(rows):
                for c in range(cols):
                    if board[r][c] == "O":
                        board[r][c] = "X"
            
            # 3. Capture all unsurrounding regions -> "T" -- "O"
            for r in range(rows):
                for c in range(cols):
                    if board[r][c] == "T":
                        board[r][c] = "O"
    ```
    
- **[994. Rotting Oranges](https://leetcode.com/problems/rotting-oranges/)**
    
    ```python
    class Solution:
        def orangesRotting(self, grid: List[List[int]]) -> int:
            # time variable and the number of fresh oranges
            time, fresh = 0, 0
            q = collections.deque()
            
            rows, cols = len(grid), len(grid[0])
            
            # count the number of fresh oranges 
            # initialize the q deque (using multi-source BFS here)
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == 1:
                        fresh += 1
                    if grid[r][c] == 2:
                        q.append([r, c])
                               
            directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
            
            while q and fresh > 0:    
                
                for i in range(len(q)):
                    row, col = q.popleft()
                    for dr, dc in directions:
                        r, c = row + dr, col + dc
                        if (r not in range(rows) or 
                            c not in range(cols) or
                            grid[r][c] != 1):
                            continue
                        grid[r][c] = 2
                        q.append([r, c])
                        fresh -= 1
                        
                time += 1
            
            return time if fresh == 0 else -1
    ```
    
- **[207. Course Schedule](https://leetcode.com/problems/course-schedule/)**
    
    ```python
    class Solution:
        def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
            # initialize a prerequisites map for each courses
            preMap = { i: [] for i in range(numCourses)}
            for crs, pre in prerequisites:
                preMap[crs].append(pre)
            
            visit = set()   # indicate whether the crs has been visited
            def dfs(crs):
                if crs in visit:
                    return False
                if preMap[crs] == []: # if no prerequisites crs, the crs could be taken
                    return True
                
                visit.add(crs)
                for pre in preMap[crs]:
                    if not dfs(pre):
                        return False
                visit.remove(crs)
                preMap[crs] = []
                return True
            
            for crs in range(numCourses):
                if not dfs(crs):
                    return False
            return True
    ```
    
- **[210. Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)**
    
    ```python
    class Solution:
        def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
            res = []
            
            preMap = {i: [] for i in range(numCourses)}
            for crs, pre in prerequisites:
                preMap[crs].append(pre)
            
            visit, cycle = set(), set()
            
            def dfs(crs):
                if crs in cycle:
                    return False
                if crs in visit:
                    return True
    
                cycle.add(crs)
                for pre in preMap[crs]:
                    if not dfs(pre):
                        return False
                cycle.remove(crs)
                visit.add(crs)
                res.append(crs)
                return True
            
            for crs in range(numCourses):
                if not dfs(crs):
                    return []
            return res
    ```
    
- **[684. Redundant Connection](https://leetcode.com/problems/redundant-connection/)**
    
    ```python
    class Solution:
        def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
            # initialize parents node (a node's parent is itself)
            par = [i for i in range(len(edges) + 1)]
            # initialize the rank
            # node with larger rank would become a parent node
            rank = [1] * (len(edges) + 1)
            
            def find(n):
                # find the root parent node (top most)
                p = par[n]
                
                while p != par[p]:  # the parent of a root parent node is itself
                    par[p] = par[par[p]]
                    p = par[p]
                
                return p
            
            def union(n1, n2):
                # connect two nodes (merge)
                p1, p2 = find(n1), find(n2)
                
                if p1 == p2:    
                    # if having the same parent node, they are already connected
                    # the edge would be redundant
                    return False
                
                if rank[p1] > rank[p2]:
                    # p1 would become p1's parent
                    par[p2] = p1
                    rank[p1] += rank[p2]
                else:
                    par[p1] = p2
                    rank[p2] += rank[p1]
                return True
            
            for n1, n2 in edges:
                if not union(n1, n2):
                    return [n1, n2]
    ```
    
- ****323. Number of Connected Components in an Undirected Graph****
    
    **Problem:**
    
    ```
    Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes), write a function to find the number of connected components in an undirected graph.
    
    Example 1:
         0          3
         |          |
         1 --- 2    4
    Given n = 5 and edges = [[0, 1], [1, 2], [3, 4]], return 2.
    
    Example 2:
         0           4
         |           |
         1 --- 2 --- 3
    Given n = 5 and edges = [[0, 1], [1, 2], [2, 3], [3, 4]], return 1.
    
    Note:
    You can assume that no duplicate edges will appear in edges. Since all edges are undirected, [0, 1] is the same as [1, 0] and thus will not appear together in edges.
    
    ```
    
    ```python
    class Solution:
    		def countComponents(self, n: int, edges: List[List[int]]) -> int:
    				# use the UnionFind method
    				par = [i for i in range(n)]
    				rank = [1] * n
    				
    				def find(n):
    				'find the root parent nodes'
    						res = n
    						
    						while res != par[res]:
    								par[res] = par[par[s]]
    								res = par[res]
    						return res
    			
    				def union(n1, n2):
    						p1, p2 = find(n1), find(n2)
    						
    						if n1 == n2:
    								# they are connected
    								return 0
    						
    						if rank[p1] > rank[p2]:
    								par[p2] = p1
    				        rank[p1] += rank[p2]
    						else:
    						    par[p1] = p2
    						    rank[p2] += rank[p1]
    						return 1 # made a union
    		
    				res = n				
    				for n1, n2 in edges:
    					  res -= union(n1, n2)
    				
    				return res
    					
    ```
    
- [**178.  Graph Valid Tree**](https://www.lintcode.com/problem/178/)
    
    ```python
    from typing import (
        List,
    )
    
    class Solution:
        """
        @param n: An integer
        @param edges: a list of undirected edges
        @return: true if it's a valid tree, or false
        """
        def valid_tree(self, n: int, edges: List[List[int]]) -> bool:
            # write your code here
            if not n:
                return True
            
            adj = {i: [] for i in range(n)}
            for n1, n2 in edges:
    						# find the neighbors for every single nodes
                adj[n1].append(n2)
                adj[n2].append(n1)
            
    				# 1. check the number of nodes we visited matches the given n
    				# 2. check if there is a cycle in the tree
            visit = set() 
    
            def dfs(i, prev):
                if i in visit:
                    return False
                
                visit.add(i)
                for j in adj[i]:
                    if j == prev:   # prev is the node we previously visited, preventing false detection
                        continue
                    if not dfs(j, i):
                        return False
                return True
            
            return dfs(0, -1) and n == len(visit)
    ```
    

## Hard

- **[127. Word Ladder](https://leetcode.com/problems/word-ladder/submissions/)**
    - Problem
        
        A **transformation sequence** from word `beginWord` to word `endWord` using a dictionary `wordList` is a sequence of words `beginWord -> s1 -> s2 -> ... -> sk` such that:
        
        - Every adjacent pair of words differs by a single letter.
        - Every `si` for `1 <= i <= k` is in `wordList`. Note that `beginWord` does not need to be in `wordList`.
        - `sk == endWord`
        
        Given two words, `beginWord` and `endWord`, and a dictionary `wordList`, return *the **number of words** in the **shortest transformation sequence** from* `beginWord` *to* `endWord`*, or* `0` *if no such sequence exists.*
        
    - Example
        
        **Example 1:**
        
        ```
        Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
        Output: 5
        Explanation: One shortest transformation sequence is "hit" -> "hot" -> "dot" -> "dog" -> cog", which is 5 words long.
        
        ```
        
        **Example 2:**
        
        ```
        Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
        Output: 0
        Explanation: The endWord "cog" is not in wordList, therefore there is no valid transformation sequence.
        ```
        
    - Solution
        
        ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2012.png)
        
    - Code
        
        ```python
        class Solution:
            def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
                if endWord not in wordList:
                    return 0
                
                # buid the adjacency list
                nei = collections.defaultdict(list)
                wordList.append(beginWord)
                for word in wordList:
                    for j in range(len(word)):
                        pattern = word[: j] + '*' + word[j + 1 :]
                        nei[pattern].append(word)
                
                # do the BFS
                visit = set() # do not visit a word twice
                q = collections.deque()
                q.append(beginWord)
                res = 1
                
                while q:
                    for i in range(len(q)):
                        word = q.popleft()
                        if word == endWord:
                            return res
                        # add the same pattern to the q
                        for j in range(len(word)):
                            pattern = word[: j] + '*' + word[j + 1 :]
                            for neiWord in nei[pattern]:
                                if neiWord not in visit:
                                    visit.add(neiWord)
                                    q.append(neiWord)
                    res += 1
                
                return 0
        ```
        

# Advanced Graphs

## Medium

- **[1584. Min Cost to Connect All Points](https://leetcode.com/problems/min-cost-to-connect-all-points/)**
    
    Prim’s Algorithm is used here with time complexity of $O(n^2log(n))$. 
    
    **Goal: want to connect all nodes together without creating a cycle with minimum costs.**
    
    1. Manually creating edges by creating an adjacency map.
    2. Use **Prim’s Algorithm** to find the minimum spanning tree.
    
    ```python
    class Solution:
        def minCostConnectPoints(self, points: List[List[int]]) -> int:
            adj = {i: [] for i in range(len(points))} # [cost, point]
            
            # construct the adjacency map
            for i in range(len(points)):
                x1, y1 = points[i]
                for j in range(i + 1, len(points)):
                    x2, y2 = points[j]
                    dist = abs(x1 - x2) + abs(y1 - y2)
                    adj[i].append([dist, j])
                    adj[j].append([dist, i])
            
            # Prime's algorithm
            minH = [[0, 0]] # [cost, point]
            visit = set()   # avoid adding multipe edges
            res = 0
            
            while len(visit) < len(points):
                cost, point = heapq.heappop(minH) # pop point with minimum cost
                if point in visit: # if the point has been visited, skip it
                    continue
                visit.add(point) # else, add the point to the visit list
                res += cost # add the cost of creating the edge to the total cost
                for neiCost, nei in adj[point]: # add neighbors of the point to the minHeap
                    heapq.heappush(minH, [neiCost, nei])
            return res
    ```
    
- **[743. Network Delay Time](https://leetcode.com/problems/network-delay-time/)**
    
    Dijkstra shortest path problem using a minHeap.
    
    Time complexity: `$O(Elog(V))$`
    
    ```python
    class Solution:
        def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
            adj = {i: [] for i in range(1, n + 1)} # [time, node]
            
            for u, v, w in times:
                adj[u].append([w, v])
            
            visit = set()
            minH = [[0, k]]
            t = 0
            
            while minH:
                w1, n1 = heapq.heappop(minH)
                if n1 in visit:
                    continue
                visit.add(n1)
                t = max(t, w1)
    
                for w2, n2 in adj[n1]:
                    if n2 not in visit:
                        heapq.heappush(minH, [w1 + w2, n2])
            
            return t if len(visit) == n else -1
    ```
    
- **[787. Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/)**
    
    Using the Bellman-Ford Algorithm with time complexity `O(E + k)`
    
    - the constraints k makes it hard to implement Dijkstra method;
    - Dijkstra cannot deal with negative weights while Bellman-Ford could
    
    Start with the source node, do (k + 1) layer of BFS while tracking the minimum price
    
    ```python
    class Solution:
        def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
            prices = [float("inf")] * n
            prices[src] = 0
            
            for i in range(k + 1):
                tempPrices = prices.copy()  # store the prices for the whole layer
                
                for s, d, p in flights:
                    if prices[s] == float("inf"):
                        continue
                    if prices[s] + p < tempPrices[d]:
                        tempPrices[d] = prices[s] + p
                prices = tempPrices
            
            return -1 if prices[dst] == float('inf') else prices[dst]
    ```
    

## Hard

- **[332. Reconstruct Itinerary](https://leetcode.com/problems/reconstruct-itinerary/)**
    - Problem
        
        You are given a list of airline `tickets` where `tickets[i] = [fromi, toi]` represent the departure and the arrival airports of one flight. Reconstruct the itinerary in order and return it.
        
        All of the tickets belong to a man who departs from `"JFK"`, thus, the itinerary must begin with `"JFK"`. If there are multiple valid itineraries, you should return the itinerary that has the smallest lexical order when read as a single string.
        
        - For example, the itinerary `["JFK", "LGA"]` has a smaller lexical order than `["JFK", "LGB"]`.
        
        You may assume all tickets form at least one valid itinerary. You must use all the tickets once and only once.
        
    - Example
        
        **Example 1:**
        
        ![https://assets.leetcode.com/uploads/2021/03/14/itinerary1-graph.jpg](https://assets.leetcode.com/uploads/2021/03/14/itinerary1-graph.jpg)
        
        ```
        Input: tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
        Output: ["JFK","MUC","LHR","SFO","SJC"]
        
        ```
        
        **Example 2:**
        
        ![https://assets.leetcode.com/uploads/2021/03/14/itinerary2-graph.jpg](https://assets.leetcode.com/uploads/2021/03/14/itinerary2-graph.jpg)
        
        ```
        Input: tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
        Output: ["JFK","ATL","JFK","SFO","ATL","SFO"]
        Explanation: Another possible reconstruction is ["JFK","SFO","ATL","JFK","ATL","SFO"] but it is larger in lexical order.
        ```
        
    - Solution
        
        ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2013.png)
        
    - Code
        
        ```python
        class Solution:
            def findItinerary(self, tickets: List[List[str]]) -> List[str]:
                # create an adjacency map
                adj = {src : [] for src, dst in tickets}
                
                tickets.sort()
                for src, dst in tickets:
                    adj[src].append(dst)
                
                res = ['JFK'] # always starting from 'JFK'
                def dfs(src):
                    if len(res) == len(tickets) + 1:
                        return True
                    if src not in adj:
                        return False
                    
                    temp = list(adj[src])
                    for i, v in enumerate(temp):
                        adj[src].pop(i)
                        res.append(v)
                        if dfs(v):
                            return True
                        adj[src].insert(i, v)   # 如果变为 .append(temp[i]),会报错
                        res.pop()
                    
                    return False
                
                dfs('JFK')
                return res
        ```
        
- **[778. Swim in Rising Water](https://leetcode.com/problems/swim-in-rising-water/)**
    - Problem
        
        You are given an `n x n` integer matrix `grid` where each value `grid[i][j]` represents the elevation at that point `(i, j)`.
        
        The rain starts to fall. At time `t`, the depth of the water everywhere is `t`. You can swim from a square to another 4-directionally adjacent square if and only if the elevation of both squares individually are at most `t`. You can swim infinite distances in zero time. Of course, you must stay within the boundaries of the grid during your swim.
        
        Return *the least time until you can reach the bottom right square* `(n - 1, n - 1)` *if you start at the top left square* `(0, 0)`.
        
    - Example
        
        **Example 1:**
        
        ![https://assets.leetcode.com/uploads/2021/06/29/swim1-grid.jpg](https://assets.leetcode.com/uploads/2021/06/29/swim1-grid.jpg)
        
        ```
        Input: grid = [[0,2],[1,3]]
        Output: 3
        Explanation:
        At time 0, you are in grid location (0, 0).
        You cannot go anywhere else because 4-directionally adjacent neighbors have a higher elevation than t = 0.
        You cannot reach point (1, 1) until time 3.
        When the depth of water is 3, we can swim anywhere inside the grid.
        
        ```
        
        **Example 2:**
        
        ![https://assets.leetcode.com/uploads/2021/06/29/swim2-grid-1.jpg](https://assets.leetcode.com/uploads/2021/06/29/swim2-grid-1.jpg)
        
        ```
        Input: grid = [[0,1,2,3,4],[24,23,22,21,5],[12,13,14,15,16],[11,17,18,19,20],[10,9,8,7,6]]
        Output: 16
        Explanation: The final route is shown.
        We need to wait until time 16 so that (0, 0) and (4, 4) are connected.
        ```
        
    - Solution
        
        ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2014.png)
        
    - Code
        
        ```python
        class Solution:
            def swimInWater(self, grid: List[List[int]]) -> int:
                N = len(grid)
                minHeap = [[grid[0][0], 0, 0]] # maxHeight, row, col
                visit = set()
                visit.add((0, 0))
                
                # 4 directions
                directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
                
                while minHeap:
                    h, row, col = heapq.heappop(minHeap)
                    visit.add((row, col))
                    
                    if row == col == N - 1:
                        return h
                    
                    for dr, dc in directions:
                        r, c = row + dr, col + dc
                        
                        if (r not in range(N) or
                            c not in range(N) or
                            (r, c) in visit):
                            continue
                        
                        visit.add((r, c))
                        heapq.heappush(minHeap, [max(h, grid[r][c]), r, c])
        ```
        
    

# **1-D Dynamic Programming**

> Frequently asked by Google!!
> 

## E**asy**

- **[70.Climbing Stairs](notion://www.notion.so/leahishere/70/Climbing%20Stairs)**
    
    ```python
    class Solution:
        def climbStairs(self, n: int) -> int:
            one, two = 1, 1,
    
            for i in range(n-1):
                temp = one
                one = one + two
                two = temp
    
            return one
    ```
    
- **[746.Min Cost Climbing Stairs](https://leetcode.com/problems/min-cost-climbing-stairs/)**
    - DP solution with two single variables
    - time O(n): iterate through the array in reverse
    - memory O(1): using the input array itself
    
    ```python
    class Solution:
        def minCostClimbingStairs(self, cost: List[int]) -> int:
            # [15, 10, 20] 0
    
            cost.append(0)  # for the computation convenience
    
            # min value of making a single jump/ double jump
            for i in range(len(cost)-3, -1, -1):
                #cost[i] = min(cost[i] + cost[i+1], cost[i] + cost[i+2])
                cost[i] += min(cost[i+1], cost[i+2])
    
            return min(cost[0], cost[1])
    ```
    

## Medium

- **[198. House Robber](https://leetcode.com/problems/house-robber/)**
    
    A great dynamic programming problem.
    
    Two options for each house: 1/ rub the house n and find the maximum from (n-1) OR 2/ do not rub the house and find the maximum from (n-1)
    
    ```python
    class Solution:
        def rob(self, nums: List[int]) -> int:
            # rob1 is the maximum of (n-2)
            # rob2 is the maximum of (n-1)
            rob1, rob2 = 0, 0
            
            # [rob1, rob2, n, n+1, n+2, ...]
            for n in nums:
                temp = max(n + rob1, rob2)
                rob1 = rob2
                rob2 = temp
            
            return rob2
    ```
    
- **[213. House Robber II](https://leetcode.com/problems/house-robber-ii/)**
    
    Implement House Robber solution twice.
    
    One for Nums list without the last house; another for Nums list without the first house.
    
    ```python
    class Solution:
        def rob(self, nums: List[int]) -> int:
    
            def helper(houses):
                rob1, rob2 = 0, 0
                
                for n in houses:
                    temp = max(n + rob1, rob2)
                    rob1 = rob2
                    rob2 = temp
                return rob2
            
            return max(nums[0], helper(nums[: -1]), helper(nums[1:]))
    ```
    
- **[5. Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)**
    - Brute force solution: check if every substring is palindromic — time complexity: `O(n · n ^ 2)`
    - There are two ways to check if a string is palindromic:
        - check from the both side
        - check from the middle point and expanding outward —> a more easier and efficient way
    
    ```python
    class Solution:
        def longestPalindrome(self, s: str) -> str:
            res = ""
            resLen = 0
            
            for i in range(len(s)):
                # odd case
                l, r= i, i
                while l >= 0 and r < len(s) and s[l] == s[r]:
                    if (r - l + 1) > resLen:
                        res = s[l: r + 1]
                        resLen = r - l + 1
                    l -= 1
                    r += 1
                
                # even case
                l, r = i, i + 1
                while l >= 0 and r < len(s) and s[l] == s[r]:
                    if (r - l + 1) > resLen:
                        res = s[l: r + 1]
                        resLen = r - l + 1
                    l -= 1
                    r += 1
                
            return res
    ```
    
- **[647. Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/)**
    
    ```python
    class Solution:
        def countSubstrings(self, s: str) -> int:
            res = []
            
            for i in range(len(s)):
                # odd length
                l, r = i, i
                while l >= 0 and r < len(s) and s[l] == s[r]:
                    res.append(s[l: r + 1])
                    l -= 1
                    r += 1
                
                l, r = i, i+1
                while l >= 0 and r < len(s) and s[l] == s[r]:
                    res.append(s[l: r + 1])
                    l -= 1
                    r += 1
            return len(res)
    ```
    
- **[91. Decode Ways](https://leetcode.com/problems/decode-ways/)**
    - DP solution
    - subproblem: How many ways to decode everything except the beginning
        - take 1 string as the beginning
        - take 2 strings as the beginning,
            - the first could not be “0”
            - if the first is “1”, no constraints for the second string
            - if the first is “2”, the second string could not exceed 6 ($\because$ range from 1~26)
    
    ```python
    class Solution:
        def numDecodings(self, s: str) -> int:
            dp = {len(s): 1}
            # use real dp
            # from back to front
            for i in range(len(s) - 1, -1, -1):
                if s[i] == "0":
                    dp[i] = 0
                else:
                    dp[i] = dp[i + 1]
                
                if (i + 1 < len(s) and (s[i] == "1" or (s[i] == "2" and s[i + 1] in "0123456"))):
                    dp[i] += dp[i + 2]
            
            return dp[0]
    ```
    
    - recursive solution
    
    ```python
    class Solution:
        def numDecodings(self, s: str) -> int:
            dp = {len(s): 1}
            
            # use recursion
            def dfs(i):
                if i in dp:
                    # reach the full length of the string
                    return dp[i]
                if s[i] == "0":
                    return 0
                
                res = dfs(i + 1)    # single number
                # double number
                if (i + 1 < len(s) and (s[i] == "1" or (s[i] == "2" and s[i + 1] in "0123456"))):
                    res += dfs(i + 2)
                dp[i] = res
                return res
            
            return dfs(0)
    ```
    
- **[322. Coin Change](https://leetcode.com/problems/coin-change/)**
    - solve in reverse order
    - time complexity `O(amount * len(coins))`
    - space complexity `O(amount)`
    
    ```python
    class Solution:
        def coinChange(self, coins: List[int], amount: int) -> int:
            # use DP-Bottom-Up to solve 
            dp = [amount + 1] * (amount + 1)
            dp[0] = 0   # take 0 coins to amount 0
            
            for a in range(1, amount + 1):
                for c in coins:
                    if a - c >= 0: 
                        dp[a] = min(dp[a], 1 + dp[a - c])
            
            return dp[amount] if dp[amount] != amount + 1 else -1
    ```
    
- **[152. Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/)**
    - Brute force solution: try every single subarray. Time complexity `O(n^2)`
    - keep tracking both maximum and minimum products
        
        ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2015.png)
        
    
    ```python
    class Solution:
        def maxProduct(self, nums: List[int]) -> int:
            res = max(nums)
            curMax, curMin = 1, 1
            
            for num in nums:
                # if current number is 0, reset max and min
                if num == 0:
                    curMax, curMin = 1, 1
                    continue
                
                temp = curMax * num
                # update current max and min
                curMax = max(temp, curMin * num, num) # if [-1, 8],
                curMin = min(temp, curMin * num, num)
                
                res = max(res, curMax)
            
            return res
    ```
    
- **[139. Word Break](https://leetcode.com/problems/word-break/)**
    
    ```python
    class Solution:
        def wordBreak(self, s: str, wordDict: List[str]) -> bool:
            dp = [False] * (len(s) + 1)
            dp[len(s)] = True   # the base case, if ever get into the last index could be break
            
            for i in range(len(s) - 1, -1, -1):
                for w in wordDict:
                    # the first half check if there is enough length to compare
                    # the second half check if the word could be segmented
                    if (i + len(w)) <= len(s) and s[i: i + len(w)] == w:
                        dp[i] = dp[i + len(w)]
                    # if at least one word could be segmented, continue
                    if dp[i]:
                        break
            
            return dp[0]
    ```
    
- **[300. Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)**
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2016.png)
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2017.png)
    
    ```python
    class Solution:
        def lengthOfLIS(self, nums: List[int]) -> int:
            LIS = [1] * len(nums) # the base case is to only include itself
            
            # go through from the back
            for i in range(len(nums) - 1, -1, -1):
                # compare values
                for j in range(i + 1, len(nums)):
                    if nums[i] < nums[j]:
                        # if satisfied
                        LIS[i] = max(LIS[i], 1 + LIS[j])
            
            return max(LIS)
    ```
    
- **[416. Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/)**
    
    ```python
    class Solution:
        def canPartition(self, nums: List[int]) -> bool:
            if sum(nums) % 2 :
                return False # if the sum of the nums is odd, could not partitioned to equal half
            
            dp = set()
            dp.add(0)
            
            target = sum(nums) // 2
            
            for i in range(len(nums) - 1, -1, -1):
                nextDP = set()
                for t in dp:
                    nextDP.add(t + nums[i]) # include the number
                    nextDP.add(t)   # do not include the number
                dp = nextDP
            return True if target in dp else False
    ```
    

# 2-D Dynamic Programming

## Medium

- **[62. Unique Paths](https://leetcode.com/problems/unique-paths/)**
    
    There is a robot on an `m x n` grid. The robot is initially located at the **top-left corner** (i.e., `grid[0][0]`). The robot tries to move to the **bottom-right corner** (i.e., `grid[m - 1][n - 1]`). The robot can only move either down or right at any point in time.
    
    Given the two integers `m` and `n`, return *the number of possible unique paths that the robot can take to reach the bottom-right corner*.
    
    The test cases are generated so that the answer will be less than or equal to `2 * 109`.
    
    ```
    # Example
    **Input:** m = 3, n = 2
    **Output:** 3
    **Explanation:** From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
    1. Right -> Down -> Down
    2. Down -> Down -> Right
    3. Down -> Right -> Down
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2018.png)
    
    ```python
    # time: O(n * m)
    # memory: O(n), the length of the row
    class Solution:
        def uniquePaths(self, m: int, n: int) -> int:
    				row = [1] * n # the bottom rows are all ones
    				
    				for i in range(m - 1):
    						# the new Row is above the bottom row
    						newRow = [1] * n
    				    # go from the second last row, since the last column always = 1
    						for j in range(n - 2, -1, -1): 
    								newRow[j] = newRow[j + 1] + row[j] # right row + value below
    						row = newRow
    				
    				return row[0]
    ```
    
- **[1143. Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/) —> very popular dynamic programming program!**
    
    Given two strings `text1` and `text2`, return *the length of their longest **common subsequence**.* If there is no **common subsequence**, return `0`.
    
    A **subsequence** of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.
    
    - For example, `"ace"` is a subsequence of `"abcde"`.
    
    A **common subsequence** of two strings is a subsequence that is common to both strings.
    
    ```
    Input: text1 = "abcde", text2 = "ace" 
    Output: 3  
    Explanation: The longest common subsequence is "ace" and its length is 3.
    ```
    
    ```
    Input: text1 = "abc", text2 = "abc"
    Output: 3
    Explanation: The longest common subsequence is "abc" and its length is 3.
    ```
    
    ```
    Input: text1 = "abc", text2 = "def"
    Output: 0
    Explanation: There is no such common subsequence, so the result is 0.
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2019.png)
    
    ```python
    # time O(n·m)
    # memory O(n·m)
    class Solution:
        def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    				dp = [[0 for j in range(len(text2) + 1] for i in range(len(text1) + 1)]
    				
    				# in reverse order
    				for i in range(len(text1) - 1, -1, -1):
    						for j in range(len(text2) - 1, -1, -1):
    								if text1[i] == text2[j]:
    										dp[i][j] = 1 + dp[i+1][j+1]
    								else:
    										dp[i][j] = max(dp[i+1][j], dp[i][j+1])
    				
    				return dp[0][0]
    ```
    
- **[309. Best Time to Buy and Sell Stock with Cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)**
    
    You are given an array `prices` where `prices[i]` is the price of a given stock on the `ith` day.
    
    Find the maximum profit you can achieve. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times) with the following restrictions:
    
    - After you sell your stock, you cannot buy stock on the next day (i.e., cooldown one day).
    
    **Note:** You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
    
    **Example 1:**
    
    ```
    Input: prices = [1,2,3,0,2]
    Output: 3
    Explanation: transactions = [buy, sell, cooldown, buy, sell]
    
    ```
    
    **Example 2:**
    
    ```
    Input: prices = [1]
    Output: 0
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2020.png)
    
    ```python
    class Solution:
        def maxProfit(self, prices: List[int]) -> int:
    				# state: Buying or Selling?
    				# i is the idx of the price array (meaning the i-th day)
    				# If Buy -> i + 1
    				# If Sell -> i + 2 (need to have a cooldown state after selling)
    
    				dp = {} # key=(i, buying) val=maxProfit [using caching]
    				
    				def dfs(i, buying):
    						if i >= len(prices):
    								return 0
    						if (i, buying) in dp:
    								return dp[(i, buying)]
    					
    						if buying:
    								buy = dfs(i + 1, not buying) - prices[i] 
    								# if buy today, could not buy next day
    								# if bought, have to subtract the price we just bought
    								cooldown = dfs(i + 1, buying) # did not spend money
    								dp[(i, buying)] = max(buy, cooldown) # take the maximum profit of the two decisions
    						else:
    								# buying is False here, so (not buying == buying = True) on (i+2)th day meaning:
    								# after sell on day i, cooldown at day i + 1, buy at day i + 2
    								sell = dfs(i + 2, not buying) + prices[i] # made some money if sold 
    								cooldown = dfs(i + 1, buying)
    								dp[(i, buying)] = max(sell, cooldown)
    						return dp[(i, buying)]
    			
    				return dfs(0, True) # can only buy on the first day, so initialize buying = True
    ```
    
- **[518. Coin Change 2](https://leetcode.com/problems/coin-change-2/)**
    
    You are given an integer array `coins` representing coins of different denominations and an integer `amount` representing a total amount of money.
    
    Return *the number of combinations that make up that amount*. If that amount of money cannot be made up by any combination of the coins, return `0`.
    
    You may assume that you have an infinite number of each kind of coin.
    
    The answer is **guaranteed** to fit into a signed **32-bit** integer.
    
    **Example 1:**
    
    ```
    Input: amount = 5, coins = [1,2,5]
    Output: 4
    Explanation: there are four ways to make up the amount:
    5=5
    5=2+2+1
    5=2+1+1+1
    5=1+1+1+1+1
    
    ```
    
    **Example 2:**
    
    ```
    Input: amount = 3, coins = [2]
    Output: 0
    Explanation: the amount of 3 cannot be made up just with coins of 2.
    
    ```
    
    **Example 3:**
    
    ```
    Input: amount = 10, coins = [10]
    Output: 1
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2021.png)
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2022.png)
    
    ```python
    # using brute force depth search solution
    class Solution:
        def change(self, amount: int, coins: List[int]) -> int:
            # dfs solution
            cache = {} # key -> (i, curSum) val -> amount of combination
            
            def dfs(i, curSum):
                if i >= len(coins):
                    return 0
                if curSum > amount:
                    return 0
                if curSum == amount:
                    return 1
                if (i, curSum) in cache:
                    return cache[(i, curSum)]
                
                cache[(i, curSum)] = dfs(i, curSum + coins[i]) + dfs(i + 1, curSum)
                return cache[(i, curSum)]
            
            return dfs(0, 0)
    ```
    
    ```python
    # using DP bottom up solution
    class Solution:
        def change(self, amount: int, coins: List[int]) -> int:
            row = [0] * (amount + 1)
            row[0] = 1
             
            for i in range(len(coins) - 1, -1, -1):
                newRow = [0] * (amount + 1)
                newRow[0] = 1
                
                for j in range(1, amount + 1):
                    if (j - coins[i]) >= 0:
                        newRow[j] = newRow[j - coins[i]] + row[j]
                    else:
                        newRow[j] = row[j]
                
                row = newRow
            
            return row[amount]
    ```
    
- **[494. Target Sum](https://leetcode.com/problems/target-sum/)**
    
    You are given an integer array `nums` and an integer `target`.
    
    You want to build an **expression** out of nums by adding one of the symbols `'+'` and `'-'` before each integer in nums and then concatenate all the integers.
    
    - For example, if `nums = [2, 1]`, you can add a `'+'` before `2` and a `'-'` before `1` and concatenate them to build the expression `"+2-1"`.
    
    Return the number of different **expressions** that you can build, which evaluates to `target`.
    
    **Example 1:**
    
    ```
    Input: nums = [1,1,1,1,1], target = 3
    Output: 5
    Explanation: There are 5 ways to assign symbols to make the sum of nums be target 3.
    -1 + 1 + 1 + 1 + 1 = 3
    +1 - 1 + 1 + 1 + 1 = 3
    +1 + 1 - 1 + 1 + 1 = 3
    +1 + 1 + 1 - 1 + 1 = 3
    +1 + 1 + 1 + 1 - 1 = 3
    
    ```
    
    **Example 2:**
    
    ```
    Input: nums = [1], target = 1
    Output: 1
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2023.png)
    
    ```python
    ### 报错原因还未找到！！！
    ### 原因找到了...思路非常非常非常正确，但是base case中的条件想错了！！值得注意！！总体还是不错的
    
    class Solution:
        def findTargetSumWays(self, nums: List[int], target: int) -> int:
            # using dfs with cache
            cache = {}
            
            def dfs(i, curSum):
                if i >= len(nums):
                    return 0
                
                if (i == len(nums) - 1) and curSum == target:
                    return 1
                else:
                    return 0
                
                if (i, curSum) in cache:
                    return cache[(i, curSum)]
    
                cache[(i, curSum)] = dfs(i + 1, curSum + nums[i]) + dfs(i + 1, curSum - nums[i])
                return cache[(i, curSum)]
            
            return dfs(0, 0)
    ```
    
    ```python
    class Solution:
        def findTargetSumWays(self, nums: List[int], target: int) -> int:
            # using dfs with cache
            cache = {}
            
            def dfs(i, curSum):
                if i == len(nums):
                    return 1 if curSum == target else 0
                if (i, curSum) in cache:
                    return cache[(i, curSum)]
    
                cache[(i, curSum)] = dfs(i + 1, curSum + nums[i]) + dfs(i + 1, curSum - nums[i])
                return cache[(i, curSum)]
            
            return dfs(0, 0)
    ```
    
- **[97. Interleaving String](https://leetcode.com/problems/interleaving-string/)**
    
    Given strings `s1`, `s2`, and `s3`, find whether `s3` is formed by an **interleaving** of `s1` and `s2`.
    
    An **interleaving** of two strings `s` and `t` is a configuration where `s` and `t` are divided into `n` and `m` **non-empty** substrings respectively, such that:
    
    - `s = s1 + s2 + ... + sn`
    - `t = t1 + t2 + ... + tm`
    - `|n - m| <= 1`
    - The **interleaving** is `s1 + t1 + s2 + t2 + s3 + t3 + ...` or `t1 + s1 + t2 + s2 + t3 + s3 + ...`
    
    **Note:** `a + b` is the concatenation of strings `a` and `b`.
    
    **Example 1:**
    
    ![https://assets.leetcode.com/uploads/2020/09/02/interleave.jpg](https://assets.leetcode.com/uploads/2020/09/02/interleave.jpg)
    
    ```
    Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
    Output: true
    Explanation: One way to obtain s3 is:
    Split s1 into s1 = "aa" + "bc" + "c", and s2 into s2 = "dbbc" + "a".
    Interleaving the two splits, we get "aa" + "dbbc" + "bc" + "a" + "c" = "aadbbcbcac".
    Since s3 can be obtained by interleaving s1 and s2, we return true.
    
    ```
    
    **Example 2:**
    
    ```
    Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbbaccc"
    Output: false
    Explanation: Notice how it is impossible to interleave s2 with any other string to obtain s3.
    
    ```
    
    **Example 3:**
    
    ```
    Input: s1 = "", s2 = "", s3 = ""
    Output: true
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2024.png)
    
    ```python
    class Solution:
        def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
            if len(s1) + len(s2) != len(s3):
                return False
            
            # dfs with caching (memoization solution)
            cache = {}
            
            def dfs(i, j):
                if i == len(s1) and j == len(s2):
                    return True
                if (i, j) in cache:
                    return cache[(i, j)]
                
                if i < len(s1) and s1[i] == s3[i + j] and dfs(i + 1, j):
                    return True
                if j < len(s2) and s2[j] == s3[i + j] and dfs(i, j + 1):
                    return True
                cache[(i, j)] = False
                
                return False
            
            return dfs(0, 0)
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2025.png)
    
    ```python
    class Solution:
        def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
            if len(s1) + len(s2) != len(s3):
                return False
            
    				# 这里只能使用memory为 m * n的矩阵，因为在outerbounds情况下，也没有固定的数值，需要从outerbound开始循环
            dp = [[False for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
            dp[len(s1)][len(s2)] = True # 注意此处s1和s2的顺序，如果上一行顺序反了，这里会报错out of range
            
            for i in range(len(s1), -1, -1):
                for j in range(len(s2), -1, -1):
                    if i < len(s1) and s1[i] == s3[i + j] and dp[i + 1][j]:
                        dp[i][j] = True
                    if j < len(s2) and s2[j] == s3[i + j] and dp[i][j + 1]:
                        dp[i][j] = True
                    
            return dp[0][0]
    ```
    

## Hard

- **[329. Longest Increasing Path in a Matrix](https://leetcode.com/problems/longest-increasing-path-in-a-matrix/)**
    - Problem
        
        Given an `m x n` integers `matrix`, return *the length of the longest increasing path in* `matrix`.
        
        From each cell, you can either move in four directions: left, right, up, or down. You **may not** move **diagonally** or move **outside the boundary** (i.e., wrap-around is not allowed).
        
    - Example
        
        **Example 1:**
        
        ![https://assets.leetcode.com/uploads/2021/01/05/grid1.jpg](https://assets.leetcode.com/uploads/2021/01/05/grid1.jpg)
        
        ```
        Input: matrix = [[9,9,4],[6,6,8],[2,1,1]]
        Output: 4
        Explanation: The longest increasing path is[1, 2, 6, 9].
        
        ```
        
        **Example 2:**
        
        ![https://assets.leetcode.com/uploads/2021/01/27/tmp-grid.jpg](https://assets.leetcode.com/uploads/2021/01/27/tmp-grid.jpg)
        
        ```
        Input: matrix = [[3,4,5],[3,2,6],[2,2,1]]
        Output: 4
        Explanation:The longest increasing path is[3, 4, 5, 6]. Moving diagonally is not allowed.
        
        ```
        
        **Example 3:**
        
        ```
        Input: matrix = [[1]]
        Output: 1
        ```
        
    - Solution
        
        ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2026.png)
        
    - Code
        
        ```python
        class Solution:
            def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
                ROWS, COLS = len(matrix), len(matrix[0])
                dp = {}  # (r, c) -> LIP
        
                def dfs(r, c, prevVal):
                    if r < 0 or r == ROWS or c < 0 or c == COLS or matrix[r][c] <= prevVal:
                        return 0
                    if (r, c) in dp:
                        return dp[(r, c)]
        
                    res = 1
                    res = max(res, 1 + dfs(r + 1, c, matrix[r][c]))
                    res = max(res, 1 + dfs(r - 1, c, matrix[r][c]))
                    res = max(res, 1 + dfs(r, c + 1, matrix[r][c]))
                    res = max(res, 1 + dfs(r, c - 1, matrix[r][c]))
                    dp[(r, c)] = res
                    return res
        
                for r in range(ROWS):
                    for c in range(COLS):
                        dfs(r, c, -1)
                return max(dp.values())
        ```
        
- **[115. Distinct Subsequences](https://leetcode.com/problems/distinct-subsequences/)**
    - Problem
        
        Given two strings `s` and `t`, return *the number of distinct subsequences of `s` which equals `t`*.
        
        A string's **subsequence** is a new string formed from the original string by deleting some (can be none) of the characters without disturbing the remaining characters' relative positions. (i.e., `"ACE"` is a subsequence of `"ABCDE"` while `"AEC"` is not).
        
        The test cases are generated so that the answer fits on a 32-bit signed integer.
        
    - Example
        
        **Example 1:**
        
        ```
        Input: s = "rabbbit", t = "rabbit"
        Output: 3
        Explanation:
        As shown below, there are 3 ways you can generate "rabbit" from S.
        rabbbitrabbbitrabbbit
        ```
        
        **Example 2:**
        
        ```
        Input: s = "babgbag", t = "bag"
        Output: 5
        Explanation:
        As shown below, there are 5 ways you can generate "bag" from S.
        babgbagbabgbagbabgbagbabgbagbabgbag
        ```
        
    - Solution
        
        ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2027.png)
        
    - Code
        
        ```python
        class Solution:
            def numDistinct(self, s: str, t: str) -> int:
                # if not s and not t:
                #     return 1
                # elif not t:
                #     return 1
                # elif not s:
                #     return 0
                
                dp = {}
                
                def dfs(i, j):
                    if j >= len(t):
                        return 1
                    if i >= len(s):
                        return 0
                    if (i, j) in dp:
                        return dp[(i, j)]
                    
                    if s[i] == t[j]:
                        dp[(i, j)] = dfs(i + 1, j + 1) + dfs(i + 1, j)
                    else:
                        dp[(i, j)] = dfs(i + 1, j)
                    
                    return dp[(i, j)]
                
                return dfs(0, 0)
        ```
        
- **[72. Edit Distance](https://leetcode.com/problems/edit-distance/)**
    - Problem
        
        Given two strings `word1` and `word2`, return *the minimum number of operations required to convert `word1` to `word2`*.
        
        You have the following three operations permitted on a word:
        
        - Insert a character
        - Delete a character
        - Replace a character
    - Example
        
        **Example 1:**
        
        ```
        Input: word1 = "horse", word2 = "ros"
        Output: 3
        Explanation:
        horse -> rorse (replace 'h' with 'r')
        rorse -> rose (remove 'r')
        rose -> ros (remove 'e')
        
        ```
        
        **Example 2:**
        
        ```
        Input: word1 = "intention", word2 = "execution"
        Output: 5
        Explanation:
        intention -> inention (remove 't')
        inention -> enention (replace 'i' with 'e')
        enention -> exention (replace 'n' with 'x')
        exention -> exection (replace 'n' with 'c')
        exection -> execution (insert 'u')
        ```
        
    - Solution
        
        ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2028.png)
        
    - Code
        
        ```python
        class Solution:
            def minDistance(self, word1: str, word2: str) -> int:
                # create a 2d matrix
                cache = [[float('inf') for i in range(len(word2) + 1)] for j in range(len(word1) + 1)]
                
                # set the value for the last row
                for i in range(len(word1) + 1):
                    cache[i][len(word2)] = len(word1) - i
                # set the value for the last column
                for j in range(len(word2) + 1):
                    cache[len(word1)][j] = len(word2) - j
                
                # bottom-up
                for i in range(len(word1) - 1, -1, -1):
                    for j in range(len(word2) - 1, -1, -1):
                        if word1[i] == word2[j]:
                            cache[i][j] = cache[i + 1][j + 1]
                        else:
                            cache[i][j] = 1 + min(cache[i + 1][j], 
                                                  cache[i][j + 1], 
                                                  cache[i + 1][j + 1])
                
                return cache[0][0]
        ```
        
- **[312. Burst Balloons](https://leetcode.com/problems/burst-balloons/)**
    - Problem
        
        You are given `n` balloons, indexed from `0` to `n - 1`. Each balloon is painted with a number on it represented by an array `nums`. You are asked to burst all the balloons.
        
        If you burst the `ith` balloon, you will get `nums[i - 1] * nums[i] * nums[i + 1]` coins. If `i - 1` or `i + 1` goes out of bounds of the array, then treat it as if there is a balloon with a `1` painted on it.
        
        Return *the maximum coins you can collect by bursting the balloons wisely*.
        
    - Examples
        
        **Example 1:**
        
        ```
        Input: nums = [3,1,5,8]
        Output: 167
        Explanation:
        nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
        coins =  3*1*5    +   3*5*8   +  1*3*8  + 1*8*1 = 167
        ```
        
        **Example 2:**
        
        ```
        Input: nums = [1,5]
        Output: 10
        ```
        
    - Solution
        
        ![FF5385CA-BCDE-4DC1-9956-EAF0049A9763.jpeg](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/FF5385CA-BCDE-4DC1-9956-EAF0049A9763.jpeg)
        
    - Code
        
        ```python
        # excess time limit
        class Solution:
            def maxCoins(self, nums: List[int]) -> int:
                # modify the nums
                nums = [1] + nums + [1]
                
                dp = {}
                
                def dfs(l, r):
                    if l > r:
                        return 0 # OutOfBoundary
                    if (l, r) in dp:
                        return dp[(l, r)]
                    
                    dp[(l, r)] = 0
                    for i in range(l, r + 1):
                        coins = nums[i] * nums[l - 1] * nums[r + 1]
                        coins += dfs(l, i - 1) + dfs(i + 1, r)
                        dp[(l, r)] = max(dp[(l, r)], coins)
                    
                    return dp[(l, r)]
            
                return dfs(1, len(nums) - 2) # we add two 1s initially
        ```
        
- **[10. Regular Expression Matching](https://leetcode.com/problems/regular-expression-matching/)**
    - Problem
        
        Given an input string `s` and a pattern `p`, implement regular expression matching with support for `'.'` and `'*'` where:
        
        - `'.'` Matches any single character.
        - `'*'` Matches zero or more of the preceding element.
        
        The matching should cover the **entire** input string (not partial).
        
    - Example
        
        **Example 1:**
        
        ```
        Input: s = "aa", p = "a"
        Output: false
        Explanation: "a" does not match the entire string "aa".
        
        ```
        
        **Example 2:**
        
        ```
        Input: s = "aa", p = "a*"
        Output: true
        Explanation: '*' means zero or more of the preceding element, 'a'. Therefore, by repeating 'a' once, it becomes "aa".
        
        ```
        
        **Example 3:**
        
        ```
        Input: s = "ab", p = ".*"
        Output: true
        Explanation: ".*" means "zero or more (*) of any character (.)".
        ```
        
    - Solution
        
        ![EF4B19E8-B6D8-405E-917D-3519B537E470.jpeg](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/EF4B19E8-B6D8-405E-917D-3519B537E470.jpeg)
        
    - Code
        
        ```python
        class Solution:
            def isMatch(self, s: str, p: str) -> bool:
                cache = {}
                
                def dfs(i, j):
                    if (i >= len(s) and
                        j >= len(p)):
                        return True            
                    if (j >= len(p)):
                        return False
                    if (i, j) in cache:
                        return cache[(i, j)]
                    
                    match = (i < len(s) and (s[i] == p[j] or p[j] == '.'))
                    if (j + 1) < len(p) and p[j + 1] == '*':
                        cache[(i, j)] = (dfs(i, j + 2) or
                                         (match and dfs(i + 1, j)))
                        return cache[(i, j)]
                    
                    if match:
                        cache[(i, j)] = dfs(i + 1, j + 1)
                        return cache[(i, j)]
                    
                    cache[(i, j)] = False
                    return cache[(i, j)]
                
                return dfs(0, 0)
        ```
        
- **[44. Wildcard Matching](https://leetcode.com/problems/wildcard-matching/)**
    - Problem
        
        Given an input string (`s`) and a pattern (`p`), implement wildcard pattern matching with support for `'?'` and `'*'` where:
        
        - `'?'` Matches any single character.
        - `'*'` Matches any sequence of characters (including the empty sequence).
        
        The matching should cover the **entire** input string (not partial).
        
    - Example
        
        **Example 1:**
        
        ```
        Input: s = "aa", p = "a"
        Output: false
        Explanation: "a" does not match the entire string "aa".
        
        ```
        
        **Example 2:**
        
        ```
        Input: s = "aa", p = "*"
        Output: true
        Explanation: '*' matches any sequence.
        
        ```
        
        **Example 3:**
        
        ```
        Input: s = "cb", p = "?a"
        Output: false
        Explanation: '?' matches 'c', but the second letter is 'a', which does not match 'b'.
        ```
        
    - Code
        
        ```python
        class Solution:
            def isMatch(self, s: str, p: str) -> bool:
                cache = {}
                
                def dfs(i, j):
                    if i >= len(s) and j >= len(p):
                        return True
                    if j >= len(p):
                        return False
                    if (i, j) in cache:
                        return cache[(i, j)]
                    
                    match = (i < len(s) and (s[i] == p[j] or p[j] == '?'))
                    
                    if match:
                        cache[(i, j)] = dfs(i + 1, j + 1)
                        return cache[(i, j)]
                    elif p[j] == '*':
                        cache[(i, j)] = (i < len(s) and dfs(i + 1, j) 
                                         or dfs(i, j + 1))
                        return cache[(i, j)]
                    cache[(i, j)] = False
                    return cache[(i, j)]
                
                return dfs(0, 0)
        ```
        

# **Greedy**

## **Easy**

- **[53.Maximum Subarray](notion://www.notion.so/leahishere/53/Maximum%20Subarray)**
    - remove negative prefix
    - t: linear time O(n)
    
    ```python
    class Solution:
        def maxSubArray(self, nums: List[int]) -> int:
            maxSub = nums[0]
            curSum = 0
    
            for n in nums:
                if curSum < 0:
                    curSum = 0
                curSum += n
                maxSub = max(maxSub, curSum)
            return maxSub
    ```
    

## Medium

- **[55. Jump Game](https://leetcode.com/problems/jump-game/)**
    
    You are given an integer array `nums`. You are initially positioned at the array's **first index**, and each element in the array represents your maximum jump length at that position.
    
    Return `true` *if you can reach the last index, or* `false` *otherwise*.
    
    **Example 1:**
    
    ```
    Input: nums = [2,3,1,1,4]
    Output: true
    Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
    
    ```
    
    **Example 2:**
    
    ```
    Input: nums = [3,2,1,0,4]
    Output: false
    Explanation: You will always arrive at index 3 no matter what. Its maximum jump length is 0, which makes it impossible to reach the last index.
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2029.png)
    
    ```python
    class Solution:
    # time complexity O(n)
    # no extra memory
        def canJump(self, nums: List[int]) -> bool:
            target = len(nums) - 1
            for i in range(len(nums) - 2, -1, -1):
                if nums[i] >= target - i:
                    target = i
                else:
                    continue
            
            return target == 0
    ```
    
- **[45. Jump Game II](https://leetcode.com/problems/jump-game-ii/)**
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2030.png)
    
    ```python
    class Solution:
        def jump(self, nums: List[int]) -> int:
            # using bfs
            l, r = 0, 0 # the pointers are initially set to 0
            res = 0
            
            while r < len(nums) - 1:
                maxJump = 0
                for i in range(l, r + 1):
                    maxJump = max(maxJump, i + nums[i])
                l = r + 1
                r = maxJump
                res += 1
            return res
    ```
    
- **[134. Gas Station](https://leetcode.com/problems/gas-station/)**
    
    There are `n` gas stations along a circular route, where the amount of gas at the `ith` station is `gas[i]`.
    
    You have a car with an unlimited gas tank and it costs `cost[i]` of gas to travel from the `ith` station to its next `(i + 1)th` station. You begin the journey with an empty tank at one of the gas stations.
    
    Given two integer arrays `gas` and `cost`, return *the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return* `-1`. If there exists a solution, it is **guaranteed** to be **unique**
    
    **Example 1:**
    
    ```
    Input: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
    Output: 3
    Explanation:
    Start at station 3 (index 3) and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
    Travel to station 4. Your tank = 4 - 1 + 5 = 8
    Travel to station 0. Your tank = 8 - 2 + 1 = 7
    Travel to station 1. Your tank = 7 - 3 + 2 = 6
    Travel to station 2. Your tank = 6 - 4 + 3 = 5
    Travel to station 3. The cost is 5. Your gas is just enough to travel back to station 3.
    Therefore, return 3 as the starting index.
    
    ```
    
    **Example 2:**
    
    ```
    Input: gas = [2,3,4], cost = [3,4,3]
    Output: -1
    Explanation:
    You can't start at station 0 or 1, as there is not enough gas to travel to the next station.
    Let's start at station 2 and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
    Travel to station 0. Your tank = 4 - 3 + 2 = 3
    Travel to station 1. Your tank = 3 - 3 + 3 = 3
    You cannot travel back to station 2, as it requires 4 unit of gas but you only have 3.
    Therefore, you can't travel around the circuit once no matter where you start.
    ```
    
    - take example 1 as illustration: netConsumption = [-2, -2, -2, +3, +3]
    - start from positive netConsumption
    - keep adding netConsumption to total, if total < 0: the position would not be the starting position.
    
    ```python
    class Solution:
        def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
            if sum(gas) < sum(cost):
                return -1
    
            total = 0
            res = 0
            
            for i in range(len(gas)):
                total += gas[i] - cost[i]
                
                if total < 0:
                    total = 0
                    res = i + 1
            return res
    ```
    
- **[846. Hand of Straights](https://leetcode.com/problems/hand-of-straights/)**
    
    Alice has some number of cards and she wants to rearrange the cards into groups so that each group is of size `groupSize`, and consists of `groupSize` consecutive cards.
    
    Given an integer array `hand` where `hand[i]` is the value written on the `ith` card and an integer `groupSize`, return `true` if she can rearrange the cards, or `false` otherwise.
    
    **Example 1:**
    
    ```
    Input: hand = [1,2,3,6,2,3,4,7,8], groupSize = 3
    Output: true
    Explanation: Alice's hand can be rearranged as [1,2,3],[2,3,4],[6,7,8]
    
    ```
    
    **Example 2:**
    
    ```
    Input: hand = [1,2,3,4,5], groupSize = 4
    Output: false
    Explanation: Alice's hand can not be rearranged into groups of 4.
    
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2031.png)
    
    ```python
    class Solution:
        def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
            if len(hand) % groupSize != 0:
                return False
            
            hashmap = {}
            # create a hashmap
            for num in hand:
                hashmap[num] = hashmap.get(num, 0) + 1
            
            # create a minHeap
            minHeap = list(hashmap.keys())
            heapq.heapify(minHeap)
            
            while minHeap:
                start = minHeap[0] # get the current minimum value
                for i in range(start, start + groupSize):
                    if i in hashmap:
                        hashmap[i] -= 1
                        if hashmap[i] == 0:
                            if i != minHeap[0]: # check if the poped value is the mininum value
                                return False 
                            else:
                                heapq.heappop(minHeap)
                    else:
                        return False
            return True
    ```
    
- **[1899. Merge Triplets to Form Target Triplet](https://leetcode.com/problems/merge-triplets-to-form-target-triplet/)**
    
    A **triplet** is an array of three integers. You are given a 2D integer array `triplets`, where `triplets[i] = [ai, bi, ci]` describes the `ith` **triplet**. You are also given an integer array `target = [x, y, z]` that describes the **triplet** you want to obtain.
    
    To obtain `target`, you may apply the following operation on `triplets` **any number** of times (possibly **zero**):
    
    - Choose two indices (**0-indexed**) `i` and `j` (`i != j`) and **update** `triplets[j]` to become `[max(ai, aj), max(bi, bj), max(ci, cj)]`.
        - For example, if `triplets[i] = [2, 5, 3]` and `triplets[j] = [1, 7, 5]`, `triplets[j]` will be updated to `[max(2, 1), max(5, 7), max(3, 5)] = [2, 7, 5]`.
    
    Return `true` *if it is possible to obtain the* `target` ***triplet*** `[x, y, z]` *as an **element** of* `triplets`*, or* `false` *otherwise*.
    
    **Example 1:**
    
    ```
    Input: triplets = [[2,5,3],[1,8,4],[1,7,5]], target = [2,7,5]
    Output: true
    Explanation: Perform the following operations:
    - Choose the first and last triplets [[2,5,3],[1,8,4],[1,7,5]]. Update the last triplet to be [max(2,1), max(5,7), max(3,5)] = [2,7,5]. triplets = [[2,5,3],[1,8,4],[2,7,5]]
    The target triplet [2,7,5] is now an element of triplets.
    
    ```
    
    **Example 2:**
    
    ```
    Input: triplets = [[3,4,5],[4,5,6]], target = [3,2,5]
    Output: false
    Explanation: It is impossible to have [3,2,5] as an element because there is no 2 in any of the triplets.
    
    ```
    
    **Example 3:**
    
    ```
    Input: triplets = [[2,5,3],[2,3,4],[1,2,5],[5,2,3]], target = [5,5,5]
    Output: true
    Explanation:Perform the following operations:
    - Choose the first and third triplets [[2,5,3],[2,3,4],[1,2,5],[5,2,3]]. Update the third triplet to be [max(2,1), max(5,2), max(3,5)] = [2,5,5]. triplets = [[2,5,3],[2,3,4],[2,5,5],[5,2,3]].
    - Choose the third and fourth triplets [[2,5,3],[2,3,4],[2,5,5],[5,2,3]]. Update the fourth triplet to be [max(2,5), max(5,2), max(5,3)] = [5,5,5]. triplets = [[2,5,3],[2,3,4],[2,5,5],[5,5,5]].
    The target triplet [5,5,5] is now an element of triplets.
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2032.png)
    
    ```python
    class Solution:
        def mergeTriplets(self, triplets: List[List[int]], target: List[int]) -> bool:
            good = set()
            
            for t in triplets:
                if t[0] > target[0] or t[1] > target[1] or t[2] > target[2]:
                    continue
                    
                # if t[0] == target[0]:
                #     good.add(t[0])
                # if t[1] == target[1]:
                #     good.add(t[1])
                # if t[2] == target[2]:
                #     good.add(t[2])
                
                for i, v in enumerate(t):
                    if v == target[i]:
                        good.add(i)
                        
            return len(good) == 3
    ```
    
- **[763. Partition Labels](https://leetcode.com/problems/partition-labels/)**
    
    You are given a string `s`. We want to partition the string into as many parts as possible so that each letter appears in at most one part.
    
    Note that the partition is done so that after concatenating all the parts in order, the resultant string should be `s`.
    
    Return *a list of integers representing the size of these parts*.
    
    **Example 1:**
    
    ```
    Input: s = "ababcbacadefegdehijhklij"
    Output: [9,7,8]
    Explanation:
    The partition is "ababcbaca", "defegde", "hijhklij".
    This is a partition so that each letter appears in at most one part.
    A partition like "ababcbacadefegde", "hijhklij" is incorrect, because it splits s into less parts.
    
    ```
    
    **Example 2:**
    
    ```
    Input: s = "eccbbbbdec"
    Output: [10]
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2033.png)
    
    ```python
    class Solution:
        def partitionLabels(self, s: str) -> List[int]:
            lastIdxMap = {} # key->char; value->lastIdx
            for i in range(len(s)):
                lastIdxMap[s[i]] = i
            
            res = []    # the final output array
            size = 0    # size of one partition
            end = 0     # the end idx of one partition
            
            for i in range(len(s)):
                size += 1
                end = max(end, lastIdxMap[s[i]])
                
                if i == end:
                    res.append(size)
                    size = 0
            
            return res
    ```
    
- **[678. Valid Parenthesis String](https://leetcode.com/problems/valid-parenthesis-string/)**
    
    Given a string `s` containing only three types of characters: `'('`, `')'` and `'*'`, return `true` *if* `s` *is **valid***.
    
    The following rules define a **valid** string:
    
    - Any left parenthesis `'('` must have a corresponding right parenthesis `')'`.
    - Any right parenthesis `')'` must have a corresponding left parenthesis `'('`.
    - Left parenthesis `'('` must go before the corresponding right parenthesis `')'`.
    - `'*'` could be treated as a single right parenthesis `')'` or a single left parenthesis `'('` or an empty string `""`.
    
    **Example 1:**
    
    ```
    Input: s = "()"
    Output: true
    
    ```
    
    **Example 2:**
    
    ```
    Input: s = "(*)"
    Output: true
    
    ```
    
    **Example 3:**
    
    ```
    Input: s = "(*))"
    Output: true
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2034.png)
    
    ```python
    class Solution:
        def checkValidString(self, s: str) -> bool:
            leftmax, leftmin = 0, 0
            
            for c in s:
                if c == "(":
                    leftmax += 1
                    leftmin += 1
                elif c == "*":
                    leftmax += 1
                    leftmin -= 1
                elif c == ")":
                    leftmax -= 1
                    leftmin -= 1
                
                if leftmin < 0:
                    leftmin = 0
                if leftmax < 0:
                    return False
            
            return leftmin == 0
    ```
    

# **Intervals**

## **Easy**

- **[920.Meeting Rooms](https://leetcode.com/problems/meeting-rooms/)**
    
    ```python
    """
    Definition of Interval:
    class Interval(object):
        def __init__(self, start, end):
            self.start = start
            self.end = end
    """
    
    class Solution:
        """
        @param intervals: an array of meeting time intervals
        @return: if a person could attend all meetings
        """
        def can_attend_meetings(self, intervals: List[Interval]) -> bool:
            # Write your code here
            intervals.sort(key=lambda i: i.start)   # sort the start value
    
            for i in range(1, len(intervals)):
                i1 = intervals[i-1]
                i2 = intervals[i]
    
                if i2.start < i1.end:
                    return False
    
            return True
    ```
    

## Medium

- **[57. Insert Interval](https://leetcode.com/problems/insert-interval/)**
    
    You are given an array of non-overlapping intervals `intervals` where `intervals[i] = [starti, endi]` represent the start and the end of the `ith` interval and `intervals` is sorted in ascending order by `starti`. You are also given an interval `newInterval = [start, end]` that represents the start and end of another interval.
    
    Insert `newInterval` into `intervals` such that `intervals` is still sorted in ascending order by `starti` and `intervals` still does not have any overlapping intervals (merge overlapping intervals if necessary).
    
    Return `intervals` *after the insertion*.
    
    **Example 1:**
    
    ```
    Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
    Output: [[1,5],[6,9]]
    
    ```
    
    **Example 2:**
    
    ```
    Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
    Output: [[1,2],[3,10],[12,16]]
    Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2035.png)
    
    ```python
    class Solution:
        def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
            res = []
            
            for i in range(len(intervals)):
                if newInterval[1] < intervals[i][0]:
                    res.append(newInterval)
                    return res + intervals[i: ]
                elif newInterval[0] > intervals[i][1]:
                    res.append(intervals[i])
                else: # is overlapping
                    newInterval = [min(newInterval[0], intervals[i][0]), 
                                   max(newInterval[1], intervals[i][1])]
            
            res.append(newInterval)
            return res
    ```
    
- **[56. Merge Intervals](https://leetcode.com/problems/merge-intervals/)**
    
    Given an array of `intervals` where `intervals[i] = [starti, endi]`, merge all overlapping intervals, and return *an array of the non-overlapping intervals that cover all the intervals in the input*.
    
    **Example 1:**
    
    ```
    Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
    Output: [[1,6],[8,10],[15,18]]
    Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].
    
    ```
    
    **Example 2:**
    
    ```
    Input: intervals = [[1,4],[4,5]]
    Output: [[1,5]]
    Explanation: Intervals [1,4] and [4,5] are considered overlapping.
    ```
    
    ```python
    class Solution:
        def merge(self, intervals: List[List[int]]) -> List[List[int]]:
            intervals.sort(key = lambda i : i[0])
            output = [intervals[0]]
            
            for start, end in intervals[1:]:
                if start <= output[-1][1]:
                    # overlap
                    output[-1] = [min(start, output[-1][0]),
                                  max(end, output[-1][1])]
                
                else:
                    output.append([start, end])
            
            return output
    ```
    
- **[435. Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/)**
    
    Given an array of intervals `intervals` where `intervals[i] = [starti, endi]`, return *the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping*.
    
    **Example 1:**
    
    ```
    Input: intervals = [[1,2],[2,3],[3,4],[1,3]]
    Output: 1
    Explanation: [1,3] can be removed and the rest of the intervals are non-overlapping.
    
    ```
    
    **Example 2:**
    
    ```
    Input: intervals = [[1,2],[1,2],[1,2]]
    Output: 2
    Explanation: You need to remove two [1,2] to make the rest of the intervals non-overlapping.
    
    ```
    
    **Example 3:**
    
    ```
    Input: intervals = [[1,2],[2,3]]
    Output: 0
    Explanation: You don't need to remove any of the intervals since they're already non-overlapping.
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2036.png)
    
    ```python
    class Solution:
        def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
            intervals.sort(key = lambda i : i[0])
            res = 0
            preEnd = intervals[0][1]
            
            for start, end in intervals[1:]:
                if start < preEnd:
                    res += 1
                    preEnd = min(preEnd, end)
                else:
                    preEnd = end
            
            return res
    ```
    

## Hard

- **[1851. Minimum Interval to Include Each Query](https://leetcode.com/problems/minimum-interval-to-include-each-query/)**
    - Problem
        
        You are given a 2D integer array `intervals`, where `intervals[i] = [lefti, righti]` describes the `ith` interval starting at `lefti` and ending at `righti` **(inclusive)**. The **size** of an interval is defined as the number of integers it contains, or more formally `righti - lefti + 1`.
        
        You are also given an integer array `queries`. The answer to the `jth` query is the **size of the smallest interval** `i` such that `lefti <= queries[j] <= righti`. If no such interval exists, the answer is `-1`.
        
        Return *an array containing the answers to the queries*.
        
    - Example
        
        **Example 1:**
        
        ```
        Input: intervals = [[1,4],[2,4],[3,6],[4,4]], queries = [2,3,4,5]
        Output: [3,3,1,4]
        Explanation: The queries are processed as follows:
        - Query = 2: The interval [2,4] is the smallest interval containing 2. The answer is 4 - 2 + 1 = 3.
        - Query = 3: The interval [2,4] is the smallest interval containing 3. The answer is 4 - 2 + 1 = 3.
        - Query = 4: The interval [4,4] is the smallest interval containing 4. The answer is 4 - 4 + 1 = 1.
        - Query = 5: The interval [3,6] is the smallest interval containing 5. The answer is 6 - 3 + 1 = 4.
        
        ```
        
        **Example 2:**
        
        ```
        Input: intervals = [[2,3],[2,5],[1,8],[20,25]], queries = [2,19,5,22]
        Output: [2,-1,4,6]
        Explanation: The queries are processed as follows:
        - Query = 2: The interval [2,3] is the smallest interval containing 2. The answer is 3 - 2 + 1 = 2.
        - Query = 19: None of the intervals contain 19. The answer is -1.
        - Query = 5: The interval [2,5] is the smallest interval containing 5. The answer is 5 - 2 + 1 = 4.
        - Query = 22: The interval [20,25] is the smallest interval containing 22. The answer is 25 - 20 + 1 = 6.
        ```
        
    - Solution
        
        ![C0FBF37F-5742-472E-AA48-A43799CF9A4D.jpeg](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/C0FBF37F-5742-472E-AA48-A43799CF9A4D.jpeg)
        
    - Code
        
        ```python
        class Solution:
            def minInterval(self, intervals: List[List[int]], queries: List[int]) -> List[int]:
                # sort the intervals based on the left value
                intervals.sort()
                
                minHeap = [] # (size of interval, right value)
                res = {} # q -> minLen
                i = 0 # keep tracking
                
                for q in sorted(queries):
                    while i < len(intervals) and intervals[i][0] <= q:
                        # adding possible intervals for q to the minHeap
                        l, r = intervals[i]
                        heapq.heappush(minHeap, (r - l + 1, r))
                        i += 1
                    
                    while minHeap and minHeap[0][1] < q: 
                        # if the right vale smaller than q, the interval is invalid
                        heapq.heappop(minHeap)
                    
                    res[q] = minHeap[0][0] if minHeap else -1
                    
                return (res[q] for q in queries) # return in order
        ```
        

# **Math & Geometry**

## **Easy**

- **[202.Happy Number](https://leetcode.com/problems/happy-number/)**
    
    ```python
    class Solution:
        def isHappy(self, n: int) -> bool:
            visit = set()
    
            while n not in visit:
                visit.add(n)
                n = self.sumSquare(n)
    
                if n == 1:
                    return True
    
            return False
    
        def sumSquare(self, n):
            sumSquare = 0
            while n:
                digit = n % 10
                digit = digit ** 2
                sumSquare += digit
                n = n // 10
            return sumSquare
    ```
    
- **[66.Plus One](https://leetcode.com/problems/plus-one/)**
    
    [My way of solving this problem]
    
    ```python
    class Solution:
        def plusOne(self, digits: List[int]) -> List[int]:
            carry = 1
    
            for i in range(len(digits)-1, -1, -1):	# loop from the reverse order
                digit = digits[i] + carry
                if digit // 10 == 1:
                    digits[i] = 0
                    carry = 1
                else:
                    digits[i] = digit
                    carry = 0
    
            if digits[0] == 0:
                return [1] + digits
            else:
                return digits
    
    ```
    

## Medium

- **[48. Rotate Image](https://leetcode.com/problems/rotate-image/)**
    
    You are given an `n x n` 2D `matrix` representing an image, rotate the image by **90** degrees (clockwise).
    
    You have to rotate the image **[in-place](https://en.wikipedia.org/wiki/In-place_algorithm)**, which means you have to modify the input 2D matrix directly. **DO NOT** allocate another 2D matrix and do the rotation.
    
    **Example 1:**
    
    ![https://assets.leetcode.com/uploads/2020/08/28/mat1.jpg](https://assets.leetcode.com/uploads/2020/08/28/mat1.jpg)
    
    ```
    Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
    Output: [[7,4,1],[8,5,2],[9,6,3]]
    
    ```
    
    **Example 2:**
    
    ![https://assets.leetcode.com/uploads/2020/08/28/mat2.jpg](https://assets.leetcode.com/uploads/2020/08/28/mat2.jpg)
    
    ```
    Input: matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
    Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2037.png)
    
    ```python
    class Solution:
        def rotate(self, matrix: List[List[int]]) -> None:
            """
            Do not return anything, modify matrix in-place instead.
            """
            l, r = 0, len(matrix) - 1
            
            while l < r:
                for i in range(r - l):
                    top, bottom = l, r
                    
                    # save the topLeft
                    topLeft = matrix[top][l + i]
                    
                    # move bottom left into top left
                    matrix[top][l + i] = matrix[bottom - i][l]
                    
                    # move bottom right into bottom left
                    matrix[bottom - i][l] = matrix[bottom][r - i]
                    
                    # move top right into bottom right
                    matrix[bottom][r - i] = matrix[top + i][r]
                    
                    # move top left into top right
                    matrix[top + i][r] = topLeft
                
                l += 1
                r -= 1
            return matrix
    ```
    
- **[54. Spiral Matrix](https://leetcode.com/problems/spiral-matrix/)**
    
    Given an `m x n` `matrix`, return *all elements of the* `matrix` *in spiral order*.
    
    **Example 1:**
    
    ![https://assets.leetcode.com/uploads/2020/11/13/spiral1.jpg](https://assets.leetcode.com/uploads/2020/11/13/spiral1.jpg)
    
    ```
    Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
    Output: [1,2,3,6,9,8,7,4,5]
    
    ```
    
    **Example 2:**
    
    ![https://assets.leetcode.com/uploads/2020/11/13/spiral.jpg](https://assets.leetcode.com/uploads/2020/11/13/spiral.jpg)
    
    ```
    Input: matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    Output: [1,2,3,4,8,12,11,10,9,5,6,7]
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2038.png)
    
    ```python
    class Solution:
        def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
            l, r = 0, len(matrix[0])
            t, b = 0, len(matrix)
            res = []
            
            while l < r and t < b:
                # add the top row to the output
                for i in range(l, r):
                    res.append(matrix[t][i])
                t += 1
                # add the right column to the output
                for i in range(t, b):
                    res.append(matrix[i][r - 1])
                r -= 1
                
                if not (l < r and t < b):
                    break
                    
                # add the bottom row to the output
                for i in range(r - 1, l - 1, -1):
                    res.append(matrix[b - 1][i])
                b -= 1
                # add the left col to the output
                for i in range(b - 1, t - 1, -1):
                    res.append(matrix[i][l])
                l += 1
            
            return res
    ```
    
- **[73. Set Matrix Zeroes](https://leetcode.com/problems/set-matrix-zeroes/)**
    
    Given an `m x n` integer matrix `matrix`, if an element is `0`, set its entire row and column to `0`'s.
    
    You must do it [in place](https://en.wikipedia.org/wiki/In-place_algorithm).
    
    **Example 1:**
    
    ![https://assets.leetcode.com/uploads/2020/08/17/mat1.jpg](https://assets.leetcode.com/uploads/2020/08/17/mat1.jpg)
    
    ```
    Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
    Output: [[1,0,1],[0,0,0],[1,0,1]]
    
    ```
    
    **Example 2:**
    
    ![https://assets.leetcode.com/uploads/2020/08/17/mat2.jpg](https://assets.leetcode.com/uploads/2020/08/17/mat2.jpg)
    
    ```
    Input: matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
    Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2039.png)
    
    ```python
    class Solution:
        def setZeroes(self, matrix: List[List[int]]) -> None:
            """
            Do not return anything, modify matrix in-place instead.
            """
            ROWS, COLS = len(matrix), len(matrix[0])
            rowZero = False
            
            for r in range(ROWS):
                for c in range(COLS):
                    if matrix[r][c] == 0:
                        matrix[0][c] = 0
                        
                        if r > 0:
                            matrix[r][0] = 0
                        else:
                            rowZero = True
            
            for r in range(1, ROWS):
                for c in range(1, COLS):
                    if matrix[0][c] == 0 or matrix[r][0] == 0:
                        matrix[r][c] = 0
            
            if matrix[0][0] == 0:
                for r in range(ROWS):
                    matrix[r][0] = 0
            
            if rowZero:
                for c in range(COLS):
                    matrix[0][c] = 0
    ```
    
- **[50. Pow(x, n)](https://leetcode.com/problems/powx-n/)**
    
    Implement [pow(x, n)](http://www.cplusplus.com/reference/valarray/pow/), which calculates `x` raised to the power `n` (i.e., `xn`).
    
    **Example 1:**
    
    ```
    Input: x = 2.00000, n = 10
    Output: 1024.00000
    
    ```
    
    **Example 2:**
    
    ```
    Input: x = 2.10000, n = 3
    Output: 9.26100
    
    ```
    
    **Example 3:**
    
    ```
    Input: x = 2.00000, n = -2
    Output: 0.25000
    Explanation: 2-2 = 1/22 = 1/4 = 0.25
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2040.png)
    
    ```python
    class Solution:
        def myPow(self, x: float, n: int) -> float:
            def helper(x, n):
                # base case
                if x == 0:
                    return 0
                if n == 0:
                    return 1
                
                res = helper(x, n // 2)
                res *= res
                return res * x if n % 2 else res
            
            
            res = helper(x, abs(n))
            return res if n > 0 else 1 / res
    ```
    
- **[43. Multiply Strings](https://leetcode.com/problems/multiply-strings/)**
    
    Given two non-negative integers `num1` and `num2` represented as strings, return the product of `num1` and `num2`, also represented as a string.
    
    **Note:** You must not use any built-in BigInteger library or convert the inputs to integer directly.
    
    **Example 1:**
    
    ```
    Input: num1 = "2", num2 = "3"
    Output: "6"
    
    ```
    
    **Example 2:**
    
    ```
    Input: num1 = "123", num2 = "456"
    Output: "56088"
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2041.png)
    
    ```python
    class Solution:
        def multiply(self, num1: str, num2: str) -> str:
            if "0" in [num1, num2]:
                return "0"
            
            res = [0] * (len(num1) + len(num2)) # create the result arry
            num1, num2 = num1[::-1], num2[::-1] # reverse the string
            
            # keep tracking the idx -> i1, i2
            for i1 in range(len(num1)):
                for i2 in range(len(num2)):
                    digit = int(num1[i1]) * int(num2[i2])
                    # digit may be 12
                    
                    # put the digit into corresponding idx
                    res[i1 + i2] += digit
                    # put the carry into next idx
                    res[i1 + i2 + 1] += (res[i1 + i2]//10)
                    # update the digit
                    res[i1 + i2] = res[i1 + i2] % 10
            
            # if the res [0, 1, 0, 0], need to remove the useless '0'
            res, beg = res[::-1], 0 # beg is the beginning idx
            while beg < len(res) and res[beg] == 0:
                beg += 1
            
            # turn res into string
            res = map(str, res[beg:])
            return "".join(res)
    ```
    
- **[2013. Detect Squares](https://leetcode.com/problems/detect-squares/)**
    
    You are given a stream of points on the X-Y plane. Design an algorithm that:
    
    - **Adds** new points from the stream into a data structure. **Duplicate** points are allowed and should be treated as different points.
    - Given a query point, **counts** the number of ways to choose three points from the data structure such that the three points and the query point form an **axis-aligned square** with **positive area**.
    
    An **axis-aligned square** is a square whose edges are all the same length and are either parallel or perpendicular to the x-axis and y-axis.
    
    Implement the `DetectSquares` class:
    
    - `DetectSquares()` Initializes the object with an empty data structure.
    - `void add(int[] point)` Adds a new point `point = [x, y]` to the data structure.
    - `int count(int[] point)` Counts the number of ways to form **axis-aligned squares** with point `point = [x, y]` as described above.
    
    **Example 1:**
    
    ![https://assets.leetcode.com/uploads/2021/09/01/image.png](https://assets.leetcode.com/uploads/2021/09/01/image.png)
    
    ```
    Input
    ["DetectSquares", "add", "add", "add", "count", "count", "add", "count"]
    [[], [[3, 10]], [[11, 2]], [[3, 2]], [[11, 10]], [[14, 8]], [[11, 2]], [[11, 10]]]
    Output
    [null, null, null, null, 1, 0, null, 2]
    
    Explanation
    DetectSquares detectSquares = new DetectSquares();
    detectSquares.add([3, 10]);
    detectSquares.add([11, 2]);
    detectSquares.add([3, 2]);
    detectSquares.count([11, 10]); // return 1. You can choose:
                                   //   - The first, second, and third points
    detectSquares.count([14, 8]);  // return 0. The query point cannot form a square with any points in the data structure.
    detectSquares.add([11, 2]);    // Adding duplicate points is allowed.
    detectSquares.count([11, 10]); // return 2. You can choose:
                                   //   - The first, second, and third points
                                   //   - The first, third, and fourth points
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2042.png)
    
    ```python
    class DetectSquares:
    
        def __init__(self):
            self.pts = []
            self.ptsCount = defaultdict(int) # hashmap to count the occurance of a point (deault = 0)
    
        def add(self, point: List[int]) -> None:
            self.pts.append(point)
            self.ptsCount[tuple(point)] += 1 # list cannot be keys, must convert to tuple first
    
        def count(self, point: List[int]) -> int:
            res = 0
            px, py = point
            # iterate through the point list to find its diagnal points
            for x, y in self.pts:
                # the height and width distance of the diagal pair must be equal to form a square
                if (abs(py - y) != abs(px - x)) or py == y or px == x:
                    continue
                res += self.ptsCount[(x, py)] * self.ptsCount[(px, y)]
        
            return res
                
    
    # Your DetectSquares object will be instantiated and called as such:
    # obj = DetectSquares()
    # obj.add(point)
    # param_2 = obj.count(point)
    ```
    

# **Bit Manipulation 位运算**

## **Easy**

- **[136. Single Number](notion://www.notion.so/leahishere/136/Single%20Number)**
    
    ```python
    class Solution:
        def singleNumber(self, nums: List[int]) -> int:
            res = 0
    
            # use the XOR operation
            for num in nums:
                res = res ^ num
    
            return res
    ```
    
- **[191.Number of 1 Bits](notion://www.notion.so/leahishere/191/Number%20of%201%20Bits)**
    
    ```python
    class Solution:
        def hammingWeight(self, n: int) -> int:
            res = 0
    
            while n:
                res += n & 1
                n = n >> 1
    
            return res
    ```
    
- **[338.Counting Bits](notion://www.notion.so/leahishere/338/Counting%20Bits)**
    
    ```python
    class Solution:
        def countBits(self, n: int) -> List[int]:
            dp = [0] * (n+1)
            offset = 1  # the highest position for power of 2 
    
            for i in range(1, n+1):
                if offset * 2 == i: # check if need to update the offset
                    offset = i
                dp[i] = 1 + dp[i - offset]
    
            return dp
    ```
    
- **[190.Reverse Bits](notion://www.notion.so/leahishere/190/Reverse%20Bits)**
    
    ```python
    class Solution:
        def reverseBits(self, n: int) -> int:
            res = 0
    
            for i in range(32):
                # get the i-th bit
                bit = (n >> i) &1
    
                # logic or it with output
                res = res | (bit << (31-i)) # shift the bit to the left
    
            return res
    ```
    
- **[268.Missing Number](https://leetcode.com/problems/missing-number/)**
    
    ```python
    class Solution:
        def missingNumber(self, nums: List[int]) -> int:
            res = len(nums)
    
            for i in range(len(nums)):
                res += (i - nums[i])
    
            return res
    ```
    
- **[371. Sum of Two Integers](https://leetcode.com/problems/sum-of-two-integers/)**
    
    Given two integers `a` and `b`, return *the sum of the two integers without using the operators* `+` *and* `-`.
    
    **Example 1:**
    
    ```
    Input: a = 1, b = 2
    Output: 3
    
    ```
    
    **Example 2:**
    
    ```
    Input: a = 2, b = 3
    Output: 5
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2043.png)
    
    - 作者给了方法的思路，但是在最后讲解代码的时候居然用的java!!!!!! 迷惑
    
    ```python
    class Solution:
        def getSum(self, a: int, b: int) -> int:
            def add(a, b):
                if not a or not b:
                    return a or b
                return add(a ^ b, (a & b) << 1)
            
            if a * b < 0:   # assume a < 0, b > 0
                if a > 0:
                    return self.getSum(b, a)
                if add(~a, 1) == b: # -a == b
                    return 0
                if add(~a, 1) < b: # -a < b
                    return add(~add(add(~a, 1), add(~b, 1)), 1) # add (-a, -b)
            
            return add(a, b)
    ```
    
- **[7. Reverse Integer](https://leetcode.com/problems/reverse-integer/)**
    
    Given a signed 32-bit integer `x`, return `x` *with its digits reversed*. If reversing `x` causes the value to go outside the signed 32-bit integer range `[-231, 231 - 1]`, then return `0`.
    
    **Assume the environment does not allow you to store 64-bit integers (signed or unsigned).**
    
    **Example 1:**
    
    ```
    Input: x = 123
    Output: 321
    
    ```
    
    **Example 2:**
    
    ```
    Input: x = -123
    Output: -321
    
    ```
    
    **Example 3:**
    
    ```
    Input: x = 120
    Output: 21
    ```
    
    ![Untitled](Leetcode-Blind-75%20ccc6226b2ee04586b7a6433806fe92f8/Untitled%2044.png)
    
    ```python
    class Solution:
        def reverse(self, x: int) -> int:
            # Interger.MAX_VALUE = 2147483647 (end with 7)
            # Interger.MIN_VALUE = -2147483648 (end with 8)
            
            MIN = -2147483648
            MAX = 2147483647
            
            res = 0
            while x:
                # take the digit and chop off the digit
                digit = int(math.fmod(x, 10))   # (python dumb) -1 % 10 = 9
                x = int(x / 10)                 # (python dumb) -1 // 10 = -1, to make sure could round to zero
                
                # make sure not overflow
                if (res > MAX // 10 or # do not want to look the last digit yet
                   (res == MAX // 10 and digit >= MAX % 10)): 
                    return 0
                if (res < MIN // 10 or 
                   (res == MIN // 10 and digit <= MIN % 10)):
                    return 0
                res = (res * 10) + digit
            return res
    ```
