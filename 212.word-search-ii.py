#
# @lc app=leetcode id=212 lang=python3
#
# [212] Word Search II
#

# @lc code=start
class TrieNode:
    def __init__(self):
        self.children = {}
        self.endOfWord = False
        self.refs = 0
    
    def addWord(self, word):
        cur = self
        cur.refs += 1
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
            cur.refs += 1
        cur.endOfWord = True

    def removeWord(self, word):
        cur = self
        cur.refs -= 1
        for c in word:
            if c in cur.children:
                cur = cur.children[c]
                cur.refs -= 1

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        root = TrieNode()
        # add word in the words to the prefix tree
        for w in words:
            root.addWord(w)
        
        rows, cols = len(board), len(board[0])
        visit, res = set(), set()

        def dfs(r, c, node, word):
            if (r not in range(rows) or
                c not in range(cols) or
                board[r][c] not in node.children or
                node.children[board[r][c]].refs < 1 or
                (r, c) in visit):
                return
            # if (r not in range(rows) or
            #     c not in range(cols) or
            #     board[r][c] not in node.children or
            #     (r, c) in visit):
            #     return
            
            visit.add((r, c))
            node = node.children[board[r][c]]
            word += board[r][c]
            if node.endOfWord:
                node.endOfWord = False
                res.add(word)
                root.removeWord(word)
            
            dfs(r + 1, c, node, word)
            dfs(r - 1, c, node, word)
            dfs(r, c + 1, node, word)
            dfs(r, c - 1, node, word)
            visit.remove((r, c))
        

        for r in range(rows):
            for c in range(cols):
                dfs(r, c, root, "")
        
        return list(res)
# @lc code=end

