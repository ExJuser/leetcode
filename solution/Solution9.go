package solution

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"strings"
	"unicode"
)

func maximalSquare(matrix [][]byte) int {
	dp := make([][]int, len(matrix))
	for i := 0; i < len(matrix); i++ {
		dp[i] = make([]int, len(matrix[i]))
	}
	ans := 0
	for i := 0; i < len(dp); i++ {
		for j := 0; j < len(dp[i]); j++ {
			if matrix[i][j] == '1' {
				dp[i][j] = 1
				ans = max(ans, dp[i][j])
			}
		}
	}
	for i := 1; i < len(dp); i++ {
		for j := 1; j < len(dp[i]); j++ {
			if dp[i][j] == 1 {
				x := min(int(math.Sqrt(float64(dp[i-1][j]))), int(math.Sqrt(float64(dp[i-1][j-1]))), int(math.Sqrt(float64(dp[i][j-1])))) + 1
				dp[i][j] = x * x
				ans = max(ans, dp[i][j])
			}
		}
	}
	return ans
}
func countSquares(matrix [][]int) (ans int) {
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[i]); j++ {
			if i != 0 && j != 0 && matrix[i][j] == 1 {
				matrix[i][j] = min(matrix[i-1][j], matrix[i-1][j-1], matrix[i][j-1]) + 1
			}
			ans += matrix[i][j]
		}
	}
	return
}

// 快速排序
func quickSort(nums []int) {
	var helper func(int, int)
	helper = func(left, right int) {
		if left >= right {
			return
		}
		pivot := nums[rand.Intn(right-left+1)+left]
		i, j := left, right
		for i <= j {
			for nums[i] < pivot {
				i++
			}
			for nums[j] > pivot {
				j--
			}
			if i <= j {
				nums[i], nums[j] = nums[j], nums[i]
				i++
				j--
			}
		}
		helper(left, j)
		helper(i, right)
	}
	helper(0, len(nums)-1)
}

func quickSort_(nums []int) {
	var helper func(left, right int)
	helper = func(left, right int) {
		if left >= right {
			return
		}
		pivot := nums[rand.Intn(right-left+1)+left]
		i, j := left, right
		for i <= j {
			for nums[i] < pivot {
				i++
			}
			for nums[j] > pivot {
				j--
			}
			if i <= j {
				nums[i], nums[j] = nums[j], nums[i]
				i++
				j--
			}
		}
		helper(left, j)
		helper(i, right)
	}
	helper(0, len(nums)-1)
}
func findKthLargest(nums []int, k int) int {
	var helper func(left, right, k int) int
	helper = func(left, right, k int) int {
		if left >= right {
			return nums[k]
		}
		pivot := nums[rand.Intn(right-left+1)+left]
		i, j := left, right
		for i <= j {
			for i <= j && nums[i] < pivot {
				i++
			}
			for i <= j && nums[j] > pivot {
				j--
			}
			if i <= j {
				nums[i], nums[j] = nums[j], nums[i]
				i++
				j--
			}
		}
		if j >= k {
			return helper(left, j, k)
		} else {
			return helper(i, right, k)
		}
	}
	return helper(0, len(nums)-1, len(nums)-k)
}

// 冒泡排序：每一轮确定一个最大值冒泡冒到最后一个为止
func bubbleSort(nums []int) {
	for i := 0; i < len(nums); i++ {
		for j := 0; j < len(nums)-i-1; j++ {
			if nums[j] > nums[j+1] {
				nums[j], nums[j+1] = nums[j+1], nums[j]
			}
		}
	}
}

// 选择排序：每一轮确定一个最小值与对应位置元素交换
func selectSort(nums []int) {
	for i := 0; i < len(nums); i++ {
		minIndex := i
		for j := i + 1; j < len(nums); j++ {
			if nums[j] < nums[minIndex] {
				minIndex = j
			}
		}
		nums[i], nums[minIndex] = nums[minIndex], nums[i]
	}
}

// 其实就是选出第len(nums)/2大的数
func majorityElement(nums []int) int {
	return findKthLargest(nums, len(nums)/2+1)
}

// 前缀积和后缀积
func productExceptSelf(nums []int) []int {
	prefix := make([]int, len(nums)+1)
	suffix := make([]int, len(nums)+1)
	prefix[0] = 1
	for i := 0; i < len(nums); i++ {
		prefix[i+1] = prefix[i] * nums[i]
	}
	suffix[len(nums)] = 1
	for i := len(nums) - 1; i >= 0; i-- {
		suffix[i] = suffix[i+1] * nums[i]
	}
	fmt.Println(prefix)
	fmt.Println(suffix)
	for i := 0; i < len(nums); i++ {
		nums[i] = prefix[i] * suffix[i+1]
	}
	return nums
}

// MinStack 双栈解法
//type MinStack struct {
//	stack, minStack []int
//}
//
//func Constructor() MinStack {
//	return MinStack{
//		stack:    make([]int, 0, 1000),
//		minStack: make([]int, 0, 1000),
//	}
//}
//
//func (this *MinStack) Push(val int) {
//	this.stack = append(this.stack, val)
//	if len(this.minStack) == 0 || val < this.minStack[len(this.minStack)-1] {
//		this.minStack = append(this.minStack, val)
//	} else {
//		this.minStack = append(this.minStack, this.minStack[len(this.minStack)-1])
//	}
//}
//
//func (this *MinStack) Pop() {
//	this.stack = this.stack[:len(this.stack)-1]
//	this.minStack = this.minStack[:len(this.minStack)-1]
//}
//
//func (this *MinStack) Top() int {
//	return this.stack[len(this.stack)-1]
//}
//
//func (this *MinStack) GetMin() int {
//	return this.minStack[len(this.minStack)-1]
//}

func maxProduct(nums []int) int {
	minDP := make([]float64, len(nums))
	maxDP := make([]float64, len(nums))
	minDP[0], maxDP[0] = float64(nums[0]), float64(nums[0])
	ans := float64(nums[0])
	for i := 1; i < len(nums); i++ {
		minDP[i] = min(minDP[i-1]*float64(nums[i]), maxDP[i-1]*float64(nums[i]), float64(nums[i]))
		maxDP[i] = max(minDP[i-1]*float64(nums[i]), maxDP[i-1]*float64(nums[i]), float64(nums[i]))
		ans = max(ans, maxDP[i], minDP[i])
	}
	return int(ans)
}

//	func partitionNode(head *ListNode, x int) *ListNode {
//		leftDummy := &ListNode{}
//		rightDummy := &ListNode{}
//		p, q := leftDummy, rightDummy
//		for cur := head; cur != nil; cur = cur.Next {
//			if cur.Val < x {
//				p.Next = &ListNode{Val: cur.Val}
//				p = p.Next
//			} else {
//				q.Next = &ListNode{Val: cur.Val}
//				q = q.Next
//			}
//		}
//		p.Next = rightDummy.Next
//		return leftDummy.Next
//	}
func partitionNode(head *ListNode, x int) *ListNode {
	dummy := &ListNode{Next: head}
	slow := dummy
	for slow != nil && slow.Next != nil && slow.Next.Val < x {
		slow = slow.Next
	}
	fast := slow
	for fast != nil && fast.Next != nil {
		if fast.Next.Val < x {
			temp := fast.Next
			fast.Next = temp.Next
			temp.Next = slow.Next
			slow.Next = temp
			slow = slow.Next
		} else {
			fast = fast.Next
		}
	}
	return dummy.Next
}

// 自顶向下的归并排序
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
		left, right := dfs(node), dfs(slow)
		dummy := &ListNode{}
		p := dummy
		for left != nil && right != nil {
			if left.Val <= right.Val {
				p.Next = &ListNode{Val: left.Val}
				left = left.Next
			} else {
				p.Next = &ListNode{Val: right.Val}
				right = right.Next
			}
			p = p.Next
		}
		if left != nil {
			p.Next = left
		} else {
			p.Next = right
		}
		return dummy.Next
	}
	return dfs(head)
}

// 链表的插入排序
func insertionSortList(head *ListNode) *ListNode {
	dummy := &ListNode{Next: head}
	pre := dummy
	for cur := head; cur != nil; {
		p := dummy
		for p.Next.Val <= cur.Val && p.Next != cur {
			p = p.Next
		}
		if p.Next != cur {
			temp := cur.Next
			pre.Next = temp
			cur.Next = p.Next
			p.Next = cur
			cur = temp
		} else {
			pre = cur
			cur = cur.Next
		}
	}
	return dummy.Next
}
func deleteDuplicates(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	for cur := head; cur.Next != nil; {
		if cur.Next.Val == cur.Val {
			cur.Next = cur.Next.Next
		} else {
			cur = cur.Next
		}
	}
	return head
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func widthOfBinaryTree(root *TreeNode) (ans int) {
	type Pair struct {
		node  *TreeNode
		index int
	}
	queue := make([]Pair, 0)
	queue = append(queue, Pair{node: root, index: 1})
	for len(queue) > 0 {
		size := len(queue)
		ans = max(ans, queue[size-1].index-queue[0].index+1)
		for i := 0; i < size; i++ {
			temp := queue[0]
			queue = queue[1:]
			if temp.node.Left != nil {
				queue = append(queue, Pair{node: temp.node.Left, index: temp.index * 2})
			}
			if temp.node.Right != nil {
				queue = append(queue, Pair{node: temp.node.Right, index: temp.index*2 + 1})
			}
		}
	}
	return
}
func countBeautifulPairs(nums []int) (ans int) {
	var gcd func(a, b int) int
	var getFirstDigit func(a int) int
	gcd = func(a, b int) int {
		for b > 0 {
			a, b = b, a%b
		}
		return a
	}
	getFirstDigit = func(a int) int {
		for a > 0 {
			if a/10 == 0 {
				return a
			}
			a = a / 10
		}
		return -1
	}
	for i := 0; i < len(nums); i++ {
		for j := i + 1; j < len(nums); j++ {
			if gcd(getFirstDigit(nums[i]), nums[j]%10) == 1 {
				ans++
			}
		}
	}
	return
}
func addStrings(num1 string, num2 string) string {
	num1Bytes, num2Bytes := []byte(num1), []byte(num2)
	resBytes := make([]byte, max(len(num1), len(num2))+1)
	i, j := len(num1)-1, len(num2)-1
	carry := 0
	for i >= 0 && j >= 0 {
		val := int(num1Bytes[i]-'0') + int(num2Bytes[j]) - '0' + carry
		carry = val / 10
		val = val % 10
		resBytes[max(i, j)+1] = byte(val + '0')
		i--
		j--
	}
	for i >= 0 {
		val := int(num1Bytes[i]-'0') + carry
		carry = val / 10
		val = val % 10
		resBytes[i+1] = byte(val + '0')
		i--
	}
	for j >= 0 {
		val := int(num2Bytes[j]-'0') + carry
		carry = val / 10
		val = val % 10
		resBytes[j+1] = byte(val + '0')
		j--
	}
	if carry > 0 {
		resBytes[0] = '1'
	} else {
		resBytes = resBytes[1:]
	}
	return string(resBytes)
}
func reverseMessage(message string) string {
	messages := make([]string, 0, len(message))
	for i := 0; i < len(message); {
		if message[i] == ' ' {
			i++
		} else {
			j := i
			for j < len(message) && message[j] != ' ' {
				j++
			}
			messages = append(messages, message[i:j])
			i = j
		}
	}
	slices.Reverse(messages)
	return strings.Join(messages, " ")
}

func lengthOfLongestSubstring(s string) int {
	mp := make(map[byte]int)
	var ans, left int
	for right := 0; right < len(s); right++ {
		mp[s[right]]++
		for len(mp) < right-left+1 {
			mp[s[left]]--
			if mp[s[left]] == 0 {
				delete(mp, s[left])
			}
			left++
		}
		ans = max(ans, right-left+1)
	}
	return ans
}

//	func temperatureTrend(temperatureA []int, temperatureB []int) int {
//		ans := 1
//		length := 1
//		for i := 1; i < len(temperatureA); i++ {
//			if (temperatureA[i]-temperatureA[i-1])*(temperatureB[i]-temperatureB[i-1]) > 0 {
//				length++
//			} else if temperatureA[i] == temperatureA[i-1] && temperatureB[i] == temperatureB[i-1] {
//				length++
//			} else {
//				length = 1
//			}
//			ans = max(ans, length)
//		}
//		return ans
//	}
func detectCapitalUse(word string) bool {
	if strings.ToLower(word) == word || strings.ToUpper(word) == word {
		return true
	} else if unicode.IsUpper(rune(word[0])) && strings.ToLower(word[1:]) == word[1:] {
		return true
	}
	return false
}

func nextGreaterElements(nums []int) []int {
	ans := make([]int, 0, len(nums))
	for i := 0; i < len(ans); i++ {
		ans[i] = -1
	}
	nums = append(nums, nums...)
	stack := make([]int, 0, len(nums))
	for i := 0; i < len(nums); i++ {
		for len(stack) > 0 && nums[stack[len(stack)-1]] < nums[i] {
			ans[stack[len(stack)-1]] = nums[i]
			stack = stack[:len(stack)-1]
		}
		index := i
		if index >= len(ans) {
			index -= len(ans)
		}
		stack = append(stack, index)
	}
	return ans
}

// 图论之梦开始的地方 200.岛屿数量
// 答案好像对了 但是居然是超出内存限制
// 问题出在标记一个地方走过不应该在出队列的时候标记 而是应该在加入队列的时候就标记
//func numIslands(grid [][]byte) (ans int) {
//	//经过每一个节点 都标记它已经经过
//	//遇到每一个没有经过的节点 答案+1 并开始广度优先搜索向四周扩散
//	used := make([][]bool, len(grid))
//	for i := 0; i < len(grid); i++ {
//		used[i] = make([]bool, len(grid[0]))
//	}
//	queue := make([][2]int, 0)
//	for i := 0; i < len(grid); i++ {
//		for j := 0; j < len(grid[i]); j++ {
//			if grid[i][j] == '1' && !used[i][j] {
//				ans++
//				queue = append(queue, [2]int{i, j})
//				used[i][j] = true
//				for len(queue) > 0 {
//					i, j := queue[0][0], queue[0][1]
//					queue = queue[1:]
//					if i-1 >= 0 && grid[i-1][j] == '1' && !used[i-1][j] {
//						queue = append(queue, [2]int{i - 1, j})
//						used[i-1][j] = true
//					}
//					if i+1 < len(grid) && grid[i+1][j] == '1' && !used[i+1][j] {
//						queue = append(queue, [2]int{i + 1, j})
//						used[i+1][j] = true
//					}
//					if j-1 >= 0 && grid[i][j-1] == '1' && !used[i][j-1] {
//						queue = append(queue, [2]int{i, j - 1})
//						used[i][j-1] = true
//					}
//					if j+1 < len(grid[i]) && grid[i][j+1] == '1' && !used[i][j+1] {
//						queue = append(queue, [2]int{i, j + 1})
//						used[i][j+1] = true
//					}
//				}
//			}
//		}
//	}
//	return ans
//}

// 代码就是从抄开始学的
func numIslands(grid [][]byte) (ans int) {
	var dfs func(i, j int)
	dfs = func(i, j int) {
		if i < 0 || j < 0 || i >= len(grid) || j >= len(grid[i]) {
			return
		}
		if grid[i][j] == '0' {
			return
		}
		grid[i][j] = '0'
		dfs(i-1, j)
		dfs(i+1, j)
		dfs(i, j-1)
		dfs(i, j+1)
	}
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] == '1' {
				ans++
				dfs(i, j)
			}
		}
	}
	return
}

// BFS版本的优化版
func numIslandsBFS(grid [][]byte) (ans int) {
	var bfs func(i, j int)
	bfs = func(i, j int) {
		if i < 0 || j < 0 || i >= len(grid) || j >= len(grid[i]) {
			return
		}
		if grid[i][j] == '0' {
			return
		}
		queue := [][2]int{{i, j}}
		grid[i][j] = '0'
		for len(queue) > 0 {
			x, y := queue[0][0], queue[0][1]
			queue = queue[1:]
			bfs(x-1, y)
			bfs(x+1, y)
			bfs(x, y-1)
			bfs(x, y+1)
		}
	}
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] != '0' {
				ans++
				bfs(i, j)
			}
		}
	}
	return
}

// 695. 岛屿的最大面积
func maxAreaOfIsland(grid [][]int) (ans int) {
	var (
		dfs  func(i, j int)
		area int
	)
	dfs = func(i, j int) {
		if i < 0 || j < 0 || i >= len(grid) || j >= len(grid[i]) {
			return
		}
		if grid[i][j] == 0 {
			return
		}
		grid[i][j] = 0
		area++
		dfs(i-1, j)
		dfs(i+1, j)
		dfs(i, j-1)
		dfs(i, j+1)
	}
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] == 1 {
				area = 0
				dfs(i, j)
				ans = max(ans, area)
			}
		}
	}
	return
}

// 一个格子对周长的贡献=4-格子周围有多少个1
func islandPerimeter(grid [][]int) (ans int) {
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] == 1 {
				ans += 4
				if i-1 >= 0 && grid[i-1][j] == 1 {
					ans -= 1
				}
				if i+1 < len(grid) && grid[i+1][j] == 1 {
					ans -= 1
				}
				if j-1 >= 0 && grid[i][j-1] == 1 {
					ans -= 1
				}
				if j+1 < len(grid[i]) && grid[i][j+1] == 1 {
					ans -= 1
				}
			}
		}
	}
	return
}

// 图论入门 797.所有可能的路径
func allPathsSourceTarget(graph [][]int) (ans [][]int) {
	used := make([]bool, len(graph))
	var dfs func(start int, path []int)
	dfs = func(start int, path []int) {
		if start == len(graph)-1 {
			ans = append(ans, append([]int{}, path...))
			return
		}
		if len(path) >= len(graph) {
			return
		}
		for _, node := range graph[start] {
			if !used[node] {
				used[node] = true
				path = append(path, node)
				dfs(node, path)
				path = path[:len(path)-1]
				used[node] = false
			}
		}
	}
	dfs(0, []int{0})
	return
}

// 当输入表示节点之间的连接关系的时候不是遍历输入
// 而是从连接关系入手
// 并查集入门题
func findCircleNum(isConnected [][]int) (ans int) {
	var dfs func(start int)
	visited := make([]bool, len(isConnected))
	dfs = func(start int) {
		visited[start] = true
		for i, conn := range isConnected[start] {
			if !visited[i] && conn == 1 {
				dfs(i)
			}
		}
	}
	for i := 0; i < len(isConnected); i++ {
		if !visited[i] {
			ans++
			dfs(i)
		}
	}
	return
}
func removeTrailingZeros(num string) string {
	return strings.TrimRight(num, "0")
}
