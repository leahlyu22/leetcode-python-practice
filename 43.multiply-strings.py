#
# @lc app=leetcode id=43 lang=python3
#
# [43] Multiply Strings
#

# @lc code=start
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        # if one of the number is 0,return 0 immediately
        if '0' in [num1, num2]:
            return '0'
            
        # create a result array
        res = [0] * (len(num1) + len(num2))
        num1, num2 = num1[::-1], num2[::-1] # reverse the two lists

        for i in range(len(num1)):
            for j in range(len(num2)):
                digit = int(num1[i]) * int(num2[j])
                res[i + j] += digit
                # put the carry into next position
                res[i + j + 1] += res[i + j] // 10
                # update value in current position
                res[i + j] = res[i + j] % 10
            
        # remove excess 0s
        res, start = res[::-1], 0
        while start < len(res) and res[start]==0:
            start += 1
        
        # convert output dtype
        res = map(str, res[start:])
        return "".join(res)
                
        
# @lc code=end

