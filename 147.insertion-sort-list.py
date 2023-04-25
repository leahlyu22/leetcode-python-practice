#
# @lc app=leetcode id=147 lang=python3
#
# [147] Insertion Sort List
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def insertionSortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # the first element is initially sorted
        dummy = ListNode(None, head)
        target = dummy.next.next
        
        while target:
            cur = dummy.next
            prev = dummy
            temp = target.next
            while cur != target:
                if cur.val < target.val:
                    prev = cur
                    cur = cur.next
                else:
                    prev.next = target
                    target.next = cur
                    while cur.next != target:
                        cur = cur.next
                    cur.next = temp
                    break
            target = temp
            
        
        return dummy.next


# @lc code=end

