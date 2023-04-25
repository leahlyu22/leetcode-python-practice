#
# @lc app=leetcode id=25 lang=python3
#
# [25] Reverse Nodes in k-Group
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        groupPrev = dummy # the last node of last group

        while True:
            # get the kth node (the last node) of the group
            kth = self.get_kth(groupPrev, k)
            if not kth:
                break
            # get the first node of the next group
            groupNext = kth.next

            # in-group reverse
            prev, cur = kth.next, groupPrev.next
            while cur!= groupNext:
                temp = cur.next
                cur.next = prev
                prev = cur
                cur = temp
            
            # out-group reverse
            temp = groupPrev.next
            groupPrev.next = kth
            groupPrev = temp
        
        return dummy.next

    def get_kth(self, cur, k):
        while k > 0 and cur:
            cur = cur.next
            k -= 1
        return cur

    



        
# @lc code=end

