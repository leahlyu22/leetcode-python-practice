#
# @lc app=leetcode id=234 lang=python3
#
# [234] Palindrome Linked List
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        # reverse a link list
        def reverseList(head):
            prev = None
            cur = head
            while cur:
                temp = cur.next
                cur.next = prev
                prev = cur
                cur = temp
            return prev

        # get the length of the list
        cnt = 0
        cur = head
        while cur:
            cnt += 1
            cur = cur.next
        
        # middle of the list
        half = cnt // 2
        cur = head
        while cur and half:
            cur = cur.next
            half -= 1
        
        if cnt % 2:
            second = reverseList(cur.next)
        else:
            second = reverseList(cur)
        
        cur1 = head
        cur2 = second
        l = cnt // 2
        while cur1 and cur2 and l:
            if cur1.val != cur2.val:
                return False
            else:
                cur1 = cur1.next
                cur2 = cur2.next
                l -= 1
        return True



        
# @lc code=end

