package hot100

import (
	"container/heap"
	"math"
	"strings"
)

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

// 739. 每日温度 单调栈 下一个更高温度出现在几天后
func dailyTemperatures(temperatures []int) []int {
	stack := make([]int, 0, len(temperatures))
	ans := make([]int, len(temperatures))
	for i := 0; i < len(temperatures); i++ {
		for len(stack) > 0 && temperatures[stack[len(stack)-1]] < temperatures[i] {
			ans[stack[len(stack)-1]] = i - stack[len(stack)-1]
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, i)
	}
	return ans
}

// 221. 最大正方形 动态规划
func maximalSquare(matrix [][]byte) int {
	dp := make([][]int, len(matrix))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(matrix[i]))
	}
	var ans int
	for i := 0; i < len(matrix); i++ {
		if matrix[i][0] == '1' {
			dp[i][0] = 1
			ans = 1
		}
	}
	for i := 0; i < len(matrix[0]); i++ {
		if matrix[0][i] == '1' {
			dp[0][i] = 1
			ans = 1
		}
	}
	for i := 1; i < len(matrix); i++ {
		for j := 1; j < len(matrix[i]); j++ {
			if matrix[i][j] == '1' {
				dp[i][j] = min(dp[i-1][j], dp[i-1][j-1], dp[i][j-1]) + 1
				ans = max(ans, dp[i][j])
			}
		}
	}
	return ans * ans
}

// 215. 数组中的第K个最大元素 堆排序做法
func findKthLargest(nums []int, k int) int {
	hp := &IntMinHeap{}
	for _, num := range nums {
		heap.Push(hp, num)
		if hp.Len() > k {
			heap.Pop(hp)
		}
	}
	return (*hp)[0]
}

// 207. 课程表 有向图内不能存在环 dfs解法
//
//	func canFinish(numCourses int, prerequisites [][]int) bool {
//		//邻接表 graph[course]代表 course 所需的前提课程
//		graph := make([][]int, numCourses)
//		for _, pre := range prerequisites {
//			course, preCourse := pre[0], pre[1]
//			graph[course] = append(graph[course], preCourse)
//		}
//
//		//状态数组 0代表未访问 1代表正在递归访问中 2代表访问完成
//		visited := make([]int, numCourses)
//
//		var dfs func(course int) bool
//		dfs = func(course int) bool {
//			//存在环
//			if visited[course] == 1 {
//				return false
//			}
//			//这条路没问题
//			if visited[course] == 2 {
//				return true
//			}
//			//标记为正在访问
//			visited[course] = 1
//			for _, c := range graph[course] {
//				if !dfs(c) {
//					return false
//				}
//			}
//			visited[course] = 2
//			return true
//		}
//		for i := 0; i < numCourses; i++ {
//			if !dfs(i) {
//				return false
//			}
//		}
//		return true
//	}
//
// 207. 课程表 有向图内不能存在环 bfs解法：拓扑排序
func canFinish(numCourses int, prerequisites [][]int) bool {
	graph := make([][]int, numCourses)
	inDegree := make(map[int]int)
	for _, pre := range prerequisites {
		course, preCourse := pre[0], pre[1]
		//邻接表：一个课程所指向(前提)的课程
		graph[course] = append(graph[course], preCourse)
		//被当做前提的课程入度++
		inDegree[preCourse]++
	}
	//找出入度为0的节点先直接加入队列
	queue := make([]int, 0, numCourses)
	for i := 0; i < numCourses; i++ {
		if inDegree[i] == 0 {
			queue = append(queue, i)
		}
	}
	var cnt int
	for len(queue) > 0 {
		course := queue[0]
		queue = queue[1:]
		//课程出队列意味着可以直接选 满足的+1
		cnt++
		for _, c := range graph[course] {
			inDegree[c]--
			if inDegree[c] == 0 {
				delete(inDegree, c)
				queue = append(queue, c)
			}
		}
	}
	return cnt == numCourses
}

// 238. 除自身以外数组的乘积
func productExceptSelf(nums []int) []int {
	prefix := make([]int, len(nums)+1)
	prefix[0] = 1
	for i := 0; i < len(nums); i++ {
		prefix[i+1] = prefix[i] * nums[i]
	}
	suffix := make([]int, len(nums)+1)
	suffix[len(nums)] = 1
	for i := len(nums) - 1; i >= 0; i-- {
		suffix[i] = suffix[i+1] * nums[i]
	}
	ans := make([]int, len(nums))
	for i := 0; i < len(nums); i++ {
		ans[i] = prefix[i] * suffix[i+1]
	}
	return ans
}

// MinStack 155. 最小栈
type MinStack struct {
	stack    []int
	minStack []int
}

func Constructor() MinStack {
	return MinStack{
		minStack: make([]int, 0),
		stack:    make([]int, 0),
	}
}

func (this *MinStack) Push(val int) {
	this.stack = append(this.stack, val)
	if len(this.minStack) == 0 || val < this.minStack[len(this.minStack)-1] {
		this.minStack = append(this.minStack, val)
	} else {
		this.minStack = append(this.minStack, this.minStack[len(this.minStack)-1])
	}
}

func (this *MinStack) Pop() {
	this.stack = this.stack[:len(this.stack)-1]
	this.minStack = this.minStack[:len(this.minStack)-1]
}

func (this *MinStack) Top() int {
	return this.stack[len(this.stack)-1]
}

func (this *MinStack) GetMin() int {
	return this.minStack[len(this.minStack)-1]
}

// 152. 乘积最大子数组
func maxProduct(nums []int) int {
	dp := make([][2]float64, len(nums))
	//最小
	dp[0][0] = float64(nums[0])
	//最大
	dp[0][1] = float64(nums[0])
	ans := float64(nums[0])
	for i := 1; i < len(nums); i++ {
		dp[i][0] = min(dp[i-1][0]*float64(nums[i]), dp[i-1][1]*float64(nums[i]), float64(nums[i]))
		dp[i][1] = max(dp[i-1][0]*float64(nums[i]), dp[i-1][1]*float64(nums[i]), float64(nums[i]))
		ans = max(ans, dp[i][1])
	}
	return int(ans)
}

func mergeList(p, q *ListNode) *ListNode {
	dummy := &ListNode{}
	l := dummy
	for p != nil && q != nil {
		if p.Val <= q.Val {
			l.Next = p
			p = p.Next
		} else {
			l.Next = q
			q = q.Next
		}
		l = l.Next
	}
	if p == nil {
		l.Next = q
	}
	if q == nil {
		l.Next = p
	}
	return dummy.Next
}

// 链表的归并排序
func sortList(head *ListNode) *ListNode {
	var dfs func(node *ListNode) *ListNode
	dfs = func(node *ListNode) *ListNode {
		if node == nil || node.Next == nil {
			return node
		}
		slow, fast := node, node
		var preSlow *ListNode
		for fast != nil && fast.Next != nil {
			preSlow = slow
			slow = slow.Next
			fast = fast.Next.Next
		}
		preSlow.Next = nil
		left := dfs(node)
		right := dfs(slow)
		return mergeList(left, right)
	}
	return dfs(head)
}

// 139. 单词拆分 动态规划
func wordBreak(s string, wordDict []string) bool {
	//dpi 到i为止的字符串能否被拼出来
	dp := make([]bool, len(s))
	for _, word := range wordDict {
		if strings.HasPrefix(s, word) {
			dp[len(word)-1] = true
		}
	}
	for i := 0; i < len(dp); i++ {
		if dp[i] {
			for _, word := range wordDict {
				if strings.HasPrefix(s[i+1:], word) {
					dp[i+len(word)] = true
				}
			}
		}
	}
	return dp[len(s)-1]
}

// 136. 只出现一次的数字 异或运算
func singleNumber(nums []int) int {
	res := nums[0]
	for i := 1; i < len(nums); i++ {
		res ^= nums[i]
	}
	return res
}

// 647. 回文子串 动态规划
func countSubstrings(s string) int {
	dp := make([][]bool, len(s))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]bool, len(s))
	}
	//初始化对角线为true
	for i := 0; i < len(s); i++ {
		dp[i][i] = true
	}
	ans := len(s)
	//回文串问题需要考虑遍历顺序
	for j := 1; j < len(dp); j++ {
		for i := 0; i < j; i++ {
			if s[i] == s[j] {
				if j-i == 1 || dp[i+1][j-1] {
					ans++
					dp[i][j] = true
				}
			}
		}
	}
	return ans
}

// 128. 最长连续序列
func longestConsecutive(nums []int) int {
	mp := make(map[int]bool, len(nums))
	for _, num := range nums {
		mp[num] = false
	}
	var ans int
	for num, visited := range mp {
		if !visited {
			pre := num - 1
			for {
				if _, ok := mp[pre]; ok {
					mp[pre] = true
					pre--
				} else {
					break
				}
			}
			ans = max(ans, num-pre)
		}
	}
	return ans
}

// 322. 零钱兑换 dfs 最少的硬币个数那肯定是优先选面值更大的 但是会超时
//
//	func coinChange(coins []int, amount int) int {
//		slices.SortFunc(coins, func(a, b int) int {
//			return b - a
//		})
//		var dfs func(index, current, count int)
//		ans := math.MaxInt
//		dfs = func(index, current, count int) {
//			if index >= len(coins) || current > amount {
//				return
//			}
//			if current == amount {
//				ans = min(ans, count)
//				return
//			}
//			//选
//			dfs(index, current+coins[index], count+1)
//			//不选
//			dfs(index+1, current, count)
//		}
//		dfs(0, 0, 0)
//		if ans == math.MaxInt {
//			return -1
//		}
//		return ans
//	}
//
// 322. 零钱兑换 完全背包问题 只看个数不看顺序 因此遍历背包和遍历物品没有先后顺序
func coinChange(coins []int, amount int) int {
	//dpi 凑成i金额所需的最少硬币个数
	dp := make([]int, amount+1)
	for i := 1; i <= amount; i++ {
		dp[i] = math.MaxInt
	}
	for i := 0; i < len(coins); i++ {
		for j := coins[i]; j <= amount; j++ {
			if dp[j-coins[i]] != math.MaxInt {
				dp[j] = min(dp[j], dp[j-coins[i]]+1)
			}
		}
	}
	if dp[amount] == math.MaxInt {
		return -1
	}
	return dp[amount]
}
