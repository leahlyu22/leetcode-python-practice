#
# @lc app=leetcode id=146 lang=python3
#
# [146] LRU Cache
#

# @lc code=start
class Node:
    def __init__(self, key, val):
        self.key, self.val = key, val
        self.prev, self.next = None, None


class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {} # key -> val
        
        self.left, self.right = Node(0, 0), Node(0, 0)
        self.left.next = self.right
        self.right.prev = self.left
    

    def insert(self, node):
        # insert to the most right
        prev = self.right.prev
        nxt = self.right
        prev.next = nxt.prev = node
        node.prev = prev
        node.next = nxt


    def remove(self, node):
        # remove the node from the list
        prev = node.prev
        nxt = node.next
        prev.next = nxt
        nxt.prev = prev


    def get(self, key: int) -> int:
        # if key in cache:
        if key in self.cache:
             # remove the node from the current position
            self.remove(self.cache[key])
            # insert the node to the most right position(most recent use)
            self.insert(self.cache[key])
            return self.cache[key].val # remember cache[key]'s value is a Node
        # if not in the cache map
        return -1
        

    def put(self, key: int, value: int) -> None:
        # put key-value pair into the cache

        # check if key in the cache first
        if key in self.cache:
            # rm from the current position
            self.remove(self.cache[key])
        # update its value and position
        self.cache[key] = Node(key, value)
        self.insert(self.cache[key])

        # check the length
        if len(self.cache) > self.capacity:
            # get the least recent usage
            lru = self.left.next
            self.remove(lru)
            del self.cache[lru.key]

        
        


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
# @lc code=end

