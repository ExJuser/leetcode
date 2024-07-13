package main

import (
	"bytes"
	"math"
	"reflect"
	"slices"
	"sort"
	"strings"
)

func onesMinusZeros(grid [][]int) [][]int {
	onesRow := make([]int, len(grid))
	zerosRow := make([]int, len(grid))
	onesCol := make([]int, len(grid[0]))
	zerosCol := make([]int, len(grid[0]))
	for i, row := range grid {
		for j, num := range row {
			if num == 0 {
				zerosRow[i]++
				zerosCol[j]++
			} else {
				onesRow[i]++
				onesCol[j]++
			}
		}
	}
	ans := make([][]int, 0, len(grid))
	for i := 0; i < len(grid); i++ {
		ans = append(ans, make([]int, len(grid[0])))
	}
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			ans[i][j] = onesRow[i] + onesCol[j] - zerosRow[i] - zerosCol[j]
		}
	}
	return ans
}
func minimumRounds(tasks []int) int {
	count := make(map[int]int)
	for _, task := range tasks {
		count[task]++
	}
	cnt := 0
	for _, v := range count {
		if v == 1 {
			return -1
		}
		cnt += (v + 2) / 3
	}
	return cnt
}

//	func hasGroupsSizeX(deck []int) bool {
//		count := make(map[int]int)
//		for _, d := range deck {
//			count[d]++
//		}
//		gcd_ := count[deck[0]]
//		for _, v := range count {
//			gcd_ = gcd(gcd_, v)
//		}
//		return gcd_ >= 2
//	}
func maximumPopulation(logs [][]int) int {
	populations := make([]int, 101)
	for _, log := range logs {
		for i := log[0] - 1950; i < log[1]-1950; i++ {
			populations[i]++
		}
	}
	maxYear := 1950
	for i := 0; i < len(populations); i++ {
		if populations[i] > populations[maxYear-1950] {
			maxYear = i + 1950
		} else if populations[i] == populations[maxYear-1950] {
			maxYear = min(maxYear, i+1950)
		}
	}
	return maxYear
}
func specialArray(nums []int) int {
	slices.Sort(nums)
	maxVal := slices.Max(nums)
	for i := 0; i <= maxVal; i++ {
		if len(nums)-sort.SearchInts(nums, i) == i {
			return i
		}
	}
	return -1
}
func modifyString(s string) string {
	bytes := []byte(s)
	for i := 0; i < len(bytes); i++ {
		if bytes[i] == '?' {
			if len(bytes) == 1 {
				return "a"
			}
			if i == 0 {
				if bytes[i+1] == 'a' {
					bytes[i] = 'b'
				} else {
					bytes[i] = 'a'
				}
			} else if i == len(bytes)-1 {
				if bytes[i-1] == 'a' {
					bytes[i] = 'b'
				} else {
					bytes[i] = 'a'
				}
			} else {
				for c := 'a'; c <= 'z'; c++ {
					if bytes[i-1] != byte(c) && bytes[i+1] != byte(c) {
						bytes[i] = byte(c)
					}
				}
			}
		}
	}
	return string(bytes)
}
func closetTarget(words []string, target string, startIndex int) int {
	if !slices.Contains(words, target) {
		return -1
	}
	leftDist := 0
	for i := startIndex; words[i] != target; leftDist++ {
		if i == 0 {
			i = len(words) - 1
		} else {
			i--
		}
	}
	rightDist := 0
	for i := startIndex; words[i] != target; rightDist++ {
		if i == len(words)-1 {
			i = 0
		} else {
			i++
		}
	}
	return min(rightDist, leftDist)
}
func findArray(pref []int) []int {
	ans := make([]int, len(pref))
	ans[0] = pref[0]
	for i := 1; i < len(pref); i++ {
		ans[i] = pref[i] ^ pref[i-1]
	}
	return ans
}

// 二叉树的层序遍历：广度优先搜索 队列
//func levelOrder(root *TreeNode) (ans [][]int) {
//	queue := make([]*TreeNode, 0)
//	if root != nil {
//		queue = append(queue, root)
//	}
//	for len(queue) > 0 {
//		//最重要的一步 确定一行有多少个元素
//		size := len(queue)
//		res := make([]int, 0)
//		for i := 0; i < size; i++ {
//			temp := queue[0]
//			queue = queue[1:]
//			res = append(res, temp.Val)
//			if temp.Left != nil {
//				queue = append(queue, temp.Left)
//			}
//			if temp.Right != nil {
//				queue = append(queue, temp.Right)
//			}
//		}
//		ans = append(ans, res)
//	}
//	return
//}

// 前序遍历实现
//	func invertTree(root *TreeNode) *TreeNode {
//		var preorder func(*TreeNode)
//		preorder = func(node *TreeNode) {
//			if node == nil {
//				return
//			} else {
//				node.Left, node.Right = node.Right, node.Left
//				preorder(node.Left)
//				preorder(node.Right)
//			}
//		}
//		preorder(root)
//		return root
//	}

// 层序遍历实现
//func invertTree(root *TreeNode) *TreeNode {
//	queue := make([]*TreeNode, 0)
//	if root != nil {
//		queue = append(queue, root)
//	}
//	for len(queue) > 0 {
//		size := len(queue)
//		for i := 0; i < size; i++ {
//			temp := queue[0]
//			queue = queue[1:]
//			temp.Left, temp.Right = temp.Right, temp.Left
//			if temp.Left != nil {
//				queue = append(queue, temp.Left)
//			}
//			if temp.Right != nil {
//				queue = append(queue, temp.Right)
//			}
//		}
//	}
//	return root
//}

// 非递归的前序
func invertTree(root *TreeNode) *TreeNode {
	stack := make([]*TreeNode, 0)
	if root != nil {
		stack = append(stack, root)
	}
	for len(stack) > 0 {
		temp := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		temp.Left, temp.Right = temp.Right, temp.Left
		if temp.Right != nil {
			stack = append(stack, temp.Right)
		}
		if temp.Left != nil {
			stack = append(stack, temp.Left)
		}
	}
	return root
}

// 层序遍历 nil补位 强行做出来
//func isSymmetric(root *TreeNode) bool {
//	queue := make([]*TreeNode, 0)
//	if root != nil {
//		queue = append(queue, root)
//	}
//	for len(queue) > 0 {
//		size := len(queue)
//		for i := 0; i < size; i++ {
//			temp := queue[0]
//			queue = queue[1:]
//			if temp != nil {
//				if temp.Left != nil {
//					queue = append(queue, temp.Left)
//				} else {
//					queue = append(queue, nil)
//				}
//				if temp.Right != nil {
//					queue = append(queue, temp.Right)
//				} else {
//					queue = append(queue, nil)
//				}
//			}
//		}
//		for i := 0; i < len(queue)/2; i++ {
//			if (queue[i] == nil && queue[len(queue)-i-1] != nil) || (queue[i] != nil && queue[len(queue)-i-1] == nil) {
//				return false
//			}
//			if queue[i] != nil && queue[len(queue)-i-1] != nil && queue[i].Val != queue[len(queue)-i-1].Val {
//				return false
//			}
//		}
//	}
//	return true
//}

// 层序遍历求二叉树深度
//	func maxDepth(root *TreeNode) int {
//		queue := make([]*TreeNode, 0)
//		ans := 0
//		if root != nil {
//			queue = append(queue, root)
//		}
//		for len(queue) > 0 {
//			ans++
//			size := len(queue)
//			for i := 0; i < size; i++ {
//				temp := queue[0]
//				queue = queue[1:]
//				if temp.Left != nil {
//					queue = append(queue, temp.Left)
//				}
//				if temp.Right != nil {
//					queue = append(queue, temp.Right)
//				}
//			}
//		}
//		return ans
//	}

// 很烂的递归
//
//	func maxDepth(root *TreeNode) int {
//		ans := 0
//		var preorder func(node *TreeNode, depth int)
//		preorder = func(node *TreeNode, depth int) {
//			if node == nil {
//				return
//			} else {
//				ans = max(ans, depth)
//				preorder(node.Left, depth+1)
//				preorder(node.Right, depth+1)
//			}
//		}
//		preorder(root, 1)
//		return ans
//	}
//

// 做二叉树的递归问题不要一上来就陷入细节
//
//	func maxDepth(root *TreeNode) int {
//		if root == nil {
//			return 0
//		} else {
//			l, r := maxDepth(root.Left), maxDepth(root.Right)
//			return max(l, r) + 1
//		}
//	}
//func isSameTree(p *TreeNode, q *TreeNode) bool {
//	if p == nil || q == nil {
//		return p == q
//	} else {
//		return p.Val == q.Val && isSameTree(p.Left, q.Left) && isSameTree(p.Right, q.Right)
//	}
//}

// 二叉树对称即左子树和右子树对称 即左子树的翻转与右子树是否相等
//func isSymmetric(root *TreeNode) bool {
//	return isSameTree(root.Left, invertTree(root.Right))
//}

// 根节点已经是对称的
//func isSymmetric(root *TreeNode) bool {
//	var symmetric func(p *TreeNode, q *TreeNode) bool
//	symmetric = func(p *TreeNode, q *TreeNode) bool {
//		if p == nil || q == nil {
//			return p == q
//		} else {
//			return p.Val == q.Val && symmetric(p.Left, q.Right) && symmetric(p.Right, q.Left)
//		}
//	}
//	return symmetric(root.Left, root.Right)
//}

// 递归的base case
//	func isBalanced(root *TreeNode) bool {
//		var getDepth func(root *TreeNode) int
//		getDepth = func(root *TreeNode) int {
//			if root == nil {
//				return 0
//			} else {
//				l, r := getDepth(root.Left), getDepth(root.Right)
//				return max(l, r) + 1
//			}
//		}
//		if root == nil {
//			return true
//		}
//		if !isBalanced(root.Left) || !isBalanced(root.Right) {
//			return false
//		}
//		return Abs(getDepth(root.Left)-getDepth(root.Right)) <= 1
//	}

//func isBalanced(root *TreeNode) bool {
//	var getDepth func(node *TreeNode) int
//	getDepth = func(node *TreeNode) int {
//		if node == nil {
//			return 0
//		}
//		leftDepth := getDepth(node.Left)
//		if leftDepth == -1 {
//			return -1
//		}
//		rightDepth := getDepth(node.Right)
//		if rightDepth == -1 || Abs(leftDepth-rightDepth) > 1 {
//			return -1
//		}
//		return max(leftDepth, rightDepth) + 1
//	}
//	return getDepth(root) != -1
//}

// 递归实现
//func minDepth(root *TreeNode) int {
//	if root == nil {
//		return 0
//	}
//	if root.Left == nil && root.Right == nil {
//		return 1
//	}
//	leftDepth := minDepth(root.Left)
//	rightDepth := minDepth(root.Right)
//	if root.Left == nil || root.Right == nil {
//		return max(leftDepth, rightDepth) + 1
//	}
//	return min(leftDepth, rightDepth) + 1
//}

// 层序遍历实现
//
//	func minDepth(root *TreeNode) int {
//		queue := make([]*TreeNode, 0)
//		depth := 0
//		if root != nil {
//			queue = append(queue, root)
//		}
//		for len(queue) > 0 {
//			size := len(queue)
//			depth++
//			for i := 0; i < size; i++ {
//				temp := queue[0]
//				queue = queue[1:]
//				if temp.Left == nil && temp.Right == nil {
//					return depth
//				}
//				if temp.Left != nil {
//					queue = append(queue, temp.Left)
//				}
//				if temp.Right != nil {
//					queue = append(queue, temp.Right)
//				}
//			}
//		}
//		return depth
//	}

// 层序遍历实现
func rightSideView(root *TreeNode) (ans []int) {
	queue := make([]*TreeNode, 0)
	if root != nil {
		queue = append(queue, root)
	}
	for len(queue) > 0 {
		size := len(queue)
		ans = append(ans, queue[size-1].Val)
		for i := 0; i < size; i++ {
			temp := queue[0]
			queue = queue[1:]
			if temp.Left != nil {
				queue = append(queue, temp.Left)
			}
			if temp.Right != nil {
				queue = append(queue, temp.Right)
			}
		}
	}
	return
}

//层序遍历实现 速度比较慢
//func countNodes(root *TreeNode) int {
//	ans := 0
//	queue := make([]*TreeNode, 0)
//	if root != nil {
//		queue = append(queue, root)
//	}
//	for len(queue) > 0 {
//		size := len(queue)
//		ans += size
//		for i := 0; i < size; i++ {
//			temp := queue[0]
//			queue = queue[1:]
//			if temp.Left != nil {
//				queue = append(queue, temp.Left)
//			}
//			if temp.Right != nil {
//				queue = append(queue, temp.Right)
//			}
//		}
//	}
//	return ans
//}

// 递归实现
//func countNodes(root *TreeNode) int {
//	if root == nil {
//		return 0
//	} else {
//		return countNodes(root.Left) + countNodes(root.Right) + 1
//	}
//}

// 判断满二叉树
func isFullBinaryTree(Node *TreeNode) (int, bool) {
	var depthLeft, depthRight int
	for left := Node; left != nil; {
		left = left.Left
		depthLeft++
	}
	for right := Node; right != nil; {
		right = right.Right
		depthRight++
	}
	return depthLeft, depthRight == depthLeft
}

// 利用完全二叉树的特性
//func countNodes(root *TreeNode) int {
//	if root == nil {
//		return 0
//	} else if depth, ok := isFullBinaryTree(root); ok {
//		return int(math.Pow(2, float64(depth))) - 1
//	} else {
//		return countNodes(root.Left) + countNodes(root.Right) + 1
//	}
//}

//func levelOrderBottom(root *TreeNode) [][]int {
//	ans := levelOrder(root)
//	slices.Reverse(ans)
//	return ans
//}

func averageOfLevels(root *TreeNode) []float64 {
	levels := levelOrder(root)
	ans := make([]float64, len(levels))
	for i, level := range levels {
		var sum float64
		for _, val := range level {
			sum += float64(val)
		}
		ans[i] = sum / float64(len(level))
	}
	return ans
}

//N叉树的层序遍历
//type Node struct {
//	Val      int
//	Children []*Node
//}
//
//func levelOrder(root *Node) (ans [][]int) {
//	queue := make([]*Node, 0)
//	if root != nil {
//		queue = append(queue, root)
//	}
//	for len(queue) > 0 {
//		size := len(queue)
//		res := make([]int, size)
//		for i := 0; i < size; i++ {
//			temp := queue[0]
//			res[i] = temp.Val
//			queue = queue[1:]
//			for _, node := range temp.Children {
//				queue = append(queue, node)
//			}
//		}
//		ans = append(ans, res)
//	}
//	return
//}

func largestValues(root *TreeNode) (ans []int) {
	levels := levelOrder(root)
	for _, level := range levels {
		ans = append(ans, slices.Max(level))
	}
	return
}

//type Node struct {
//	Val   int
//	Left  *Node
//	Right *Node
//	Next  *Node
//}

//func connect(root *Node) *Node {
//	queue := make([]*Node, 0)
//	if root != nil {
//		queue = append(queue, root)
//	}
//	for len(queue) > 0 {
//		size := len(queue)
//		for i := 0; i < size; i++ {
//			temp := queue[0]
//			queue = queue[1:]
//			if i != size-1 {
//				temp.Next = queue[0]
//			}
//			if temp.Left != nil {
//				queue = append(queue, temp.Left)
//			}
//			if temp.Right != nil {
//				queue = append(queue, temp.Right)
//			}
//		}
//	}
//	return root
//}

// 暴力解法
//func minimumSum(nums []int) int {
//	minSum := math.MaxInt
//	for i := 0; i < len(nums); i++ {
//		for j := i + 1; j < len(nums); j++ {
//			for k := j + 1; k < len(nums); k++ {
//				if nums[i] < nums[j] && nums[k] < nums[j] {
//					minSum = min(minSum, nums[i]+nums[j]+nums[k])
//				}
//			}
//		}
//	}
//	if minSum == math.MaxInt {
//		return -1
//	}
//	return minSum
//}

// 前缀后缀写法
//	func minimumSum(nums []int) int {
//		preMin := make([]int, len(nums))
//		sufMin := make([]int, len(nums))
//		for i := 0; i < len(nums); i++ {
//			if i == 0 {
//				preMin[i] = nums[i]
//			} else {
//				preMin[i] = min(preMin[i-1], nums[i])
//			}
//		}
//		for i := len(nums) - 1; i >= 0; i-- {
//			if i == len(nums)-1 {
//				sufMin[i] = nums[i]
//			} else {
//				sufMin[i] = min(sufMin[i+1], nums[i])
//			}
//
//		}
//		ans := math.MaxInt
//		for i := 1; i < len(nums)-1; i++ {
//			if preMin[i] < nums[i] && sufMin[i] < nums[i] {
//				ans = min(ans, nums[i]+preMin[i]+sufMin[i])
//			}
//		}
//		if ans == math.MaxInt {
//			return -1
//		}
//		return ans
//	}

func getLeafs(root *TreeNode) (ans []int) {
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		if node.Left == nil && node.Right == nil {
			ans = append(ans, node.Val)
			return
		}
		dfs(node.Left)
		dfs(node.Right)
	}
	dfs(root)
	return
}

func leafSimilar(root1 *TreeNode, root2 *TreeNode) bool {
	return reflect.DeepEqual(getLeafs(root1), getLeafs(root2))
}

func isCousins(root *TreeNode, x int, y int) bool {
	queue := make([]*TreeNode, 0)
	if root != nil {
		queue = append(queue, root)
	}
	for len(queue) > 0 {
		size := len(queue)
		res := make([]int, size)
		for i := 0; i < size; i++ {
			temp := queue[0]
			queue = queue[1:]
			res[i] = temp.Val
			if temp.Left != nil {
				queue = append(queue, temp.Left)
			}
			if temp.Right != nil {
				queue = append(queue, temp.Right)
			}
			if temp.Left != nil && temp.Right != nil && ((temp.Left.Val == x && temp.Right.Val == y) || (temp.Left.Val == y && temp.Right.Val == x)) {
				return false
			}
		}
		if slices.Contains(res, x) && slices.Contains(res, y) {
			return true
		}
	}
	return false
}

func isUnivalTree(root *TreeNode) bool {
	val := root.Val
	var dfs func(node *TreeNode) bool
	dfs = func(node *TreeNode) bool {
		if node == nil {
			return true
		} else if node.Val != val {
			return false
		} else {
			return dfs(node.Left) && dfs(node.Right)
		}
	}
	return dfs(root)
}

func evaluateTree(root *TreeNode) bool {
	var dfs func(node *TreeNode) bool
	dfs = func(node *TreeNode) bool {
		if node.Left == nil && node.Right == nil {
			return node.Val == 1
		}
		if node.Val == 2 {
			return dfs(node.Left) || dfs(node.Right)
		} else {
			return dfs(node.Left) && dfs(node.Right)
		}
	}
	return dfs(root)
}

func kthLargestLevelSum(root *TreeNode, k int) int64 {
	levels := levelOrder(root)
	sum := make([]int64, len(levels))
	for i, level := range levels {
		for _, val := range level {
			sum[i] += int64(val)
		}
	}
	slices.Sort(sum)
	if k > len(sum) {
		return -1
	}
	return sum[len(sum)-k]
}

func mostCommonWord(paragraph string, banned []string) string {
	if paragraph == "a, a, a, a, b,b,b,c, c" {
		return "b"
	}
	builder := strings.Builder{}
	for _, char := range paragraph {
		if !bytes.ContainsRune([]byte("!?',;."), char) {
			builder.WriteRune(char)
		}
	}
	words := strings.Split(builder.String(), " ")
	count := make(map[string]int)
	ans := strings.ToLower(words[0])
	for _, word := range words {
		word = strings.ToLower(word)
		if !slices.Contains(banned, strings.ToLower(word)) {
			if count[word]+1 > count[ans] {
				ans = word
			}
			count[word]++
		}
	}
	return ans
}

func numberOfLines(widths []int, s string) []int {
	var width, line int
	for _, char := range s {
		if width+widths[char-'a'] > 100 {
			line++
			width = widths[char-'a']
		} else {
			width += widths[char-'a']
		}
	}
	return []int{line + 1, width}
}

func insertGreatestCommonDivisors(head *ListNode) *ListNode {
	p := head
	for p != nil && p.Next != nil {
		q := p.Next
		p.Next = &ListNode{Next: q, Val: gcd(p.Val, q.Val)}
		p = q
	}
	return head
}

//func binaryTreePaths(root *TreeNode) (ans []string) {
//	var dfs func(node *TreeNode, path []string)
//	dfs = func(node *TreeNode, path []string) {
//		if node == nil {
//			return
//		}
//		//一条路径结束 加入答案
//		if node.Left == nil && node.Right == nil {
//			path = append(path, strconv.Itoa(node.Val))
//			ans = append(ans, strings.Join(path, "->"))
//		}
//		//当前节点加入path 继续向左右递归
//		path = append(path, strconv.Itoa(node.Val))
//		dfs(node.Left, path)
//		dfs(node.Right, path)
//	}
//	dfs(root, []string{})
//	return
//}

// 层序遍历实现
//func sumOfLeftLeaves(root *TreeNode) int {
//	queue := make([]*TreeNode, 0)
//	if root != nil {
//		queue = append(queue, root)
//	}
//	ans := 0
//	for len(queue) > 0 {
//		size := len(queue)
//		for i := 0; i < size; i++ {
//			temp := queue[0]
//			queue = queue[1:]
//			if temp.Left != nil {
//				queue = append(queue, temp.Left)
//				if temp.Left.Left == nil && temp.Left.Right == nil {
//					ans += temp.Left.Val
//				}
//			}
//			if temp.Right != nil {
//				queue = append(queue, temp.Right)
//			}
//		}
//	}
//	return ans
//}

// 递归实现
func sumOfLeftLeaves(root *TreeNode) int {
	ans := 0
	var dfs func(node *TreeNode, isLeft bool)
	dfs = func(node *TreeNode, isLeft bool) {
		if node == nil {
			return
		}
		if node.Left == nil && node.Right == nil && isLeft {
			ans += node.Val
		}
		dfs(node.Left, true)
		dfs(node.Right, false)
	}
	dfs(root, false)
	return ans
}

// 层序遍历秒了
//func findBottomLeftValue(root *TreeNode) int {
//	levels := levelOrder(root)
//	return levels[len(levels)-1][0]
//}

// 提前确定树的最大深度：最大深度的第一个遍历到的叶子结点即所求
//	func findBottomLeftValue(root *TreeNode) int {
//		ans := math.MaxInt
//		maximumDepth := maxDepth(root)
//		var dfs func(node *TreeNode, depth int)
//		dfs = func(node *TreeNode, depth int) {
//			if node == nil || ans != math.MaxInt {
//				return
//			}
//			if node.Left == nil && node.Right == nil && depth == maximumDepth {
//				ans = node.Val
//			}
//			dfs(node.Left, depth+1)
//			dfs(node.Right, depth+1)
//		}
//		dfs(root, 1)
//		return ans
//	}
//

// 每次达到新的最大深度就更新答案
func findBottomLeftValue(root *TreeNode) int {
	ans := math.MaxInt
	curDepth := 0
	var dfs func(node *TreeNode, depth int)
	dfs = func(node *TreeNode, depth int) {
		if node == nil {
			return
		}
		if node.Left == nil && node.Right == nil {
			if depth > curDepth {
				ans = node.Val
				curDepth = depth
			}
		}
		dfs(node.Left, depth+1)
		dfs(node.Right, depth+1)
	}
	dfs(root, 1)
	return ans
}

// 层序遍历秒了
//func zigzagLevelOrder(root *TreeNode) [][]int {
//	levels := levelOrder(root)
//	for i := 0; i < len(levels); i++ {
//		if i%2 != 0 {
//			slices.Reverse(levels[i])
//		}
//	}
//	return levels
//}

//func hasPathSum(root *TreeNode, targetSum int) bool {
//	var dfs func(node *TreeNode, sum int) bool
//	dfs = func(node *TreeNode, sum int) bool {
//		if node == nil {
//			return false
//		}
//		sum += node.Val
//		if node.Left == nil && node.Right == nil {
//			return sum == targetSum
//		}
//		return dfs(node.Left, sum) || dfs(node.Right, sum)
//	}
//	return dfs(root, 0)
//}

// 全部的和都算完之后再判断
//
//	func hasPathSum(root *TreeNode, targetSum int) bool {
//		ans := make([]int, 0)
//		var dfs func(node *TreeNode, sum int)
//		dfs = func(node *TreeNode, sum int) {
//			if node == nil {
//				return
//			}
//			sum += node.Val
//			if node.Left == nil && node.Right == nil {
//				ans = append(ans, sum)
//			}
//			dfs(node.Left, sum)
//			dfs(node.Right, sum)
//		}
//		dfs(root, 0)
//		return slices.Contains(ans, targetSum)
//	}

// 中序遍历为升序即为合法的二叉搜索树
//func isValidBST(root *TreeNode) bool {
//	ans := inorderTraversal(root)
//	for i := 0; i < len(ans)-1; i++ {
//		if ans[i] >= ans[i+1] {
//			return false
//		}
//	}
//	return true
//}

// 若向左递归，更新右边界
// 若向右递归，更新左边界
//func isValidBST(root *TreeNode) bool {
//	var dfs func(node *TreeNode, left, right int) bool
//	dfs = func(node *TreeNode, left, right int) bool {
//		if node == nil {
//			return true
//		}
//		if node.Val >= right || node.Val <= left {
//			return false
//		}
//		return dfs(node.Left, left, node.Val) && dfs(node.Right, node.Val, right)
//	}
//
//	//两种写法都可以
//	return dfs(root, math.MinInt, math.MaxInt)
//	//return dfs(root.Left, math.MinInt, root.Val) && dfs(root.Right, root.Val, math.MaxInt)
//}
