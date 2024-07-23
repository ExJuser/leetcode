package hot100

// 160. 相交链表
func getIntersectionNode(headA, headB *ListNode) *ListNode {
	set := make(map[*ListNode]struct{})
	for p := headA; p != nil; p = p.Next {
		set[p] = struct{}{}
	}
	for p := headB; p != nil; p = p.Next {
		if _, ok := set[p]; ok {
			return p
		}
	}
	return nil
}

// 236. 二叉树的最近公共祖先
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	var dfs func(node *TreeNode) *TreeNode
	dfs = func(node *TreeNode) *TreeNode {
		if node == nil || node == p || node == q {
			return node
		}
		left := dfs(node.Left)
		right := dfs(node.Right)
		if left != nil && right != nil {
			return node
		}
		if left == nil {
			return right
		}
		return left

	}
	return dfs(root)
}
func reverseList(head *ListNode) *ListNode {
	var pre *ListNode
	for cur := head; cur != nil; {
		nxt := cur.Next
		cur.Next = pre
		cur, pre = nxt, cur
	}
	return pre
}

// 234. 回文链表 找到中点 反转 判断
func isPalindrome(head *ListNode) bool {
	//先找到中点
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	half := reverseList(slow)
	for p, q := head, half; p != nil && q != nil; p, q = p.Next, q.Next {
		if p.Val != q.Val {
			return false
		}
	}
	return true
}

// 226. 翻转二叉树
func invertTree(root *TreeNode) *TreeNode {
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		node.Left, node.Right = node.Right, node.Left
		dfs(node.Left)
		dfs(node.Right)
	}
	dfs(root)
	return root
}
