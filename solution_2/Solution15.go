package main

import (
	"math"
	"slices"
	"strconv"
	"strings"
)

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func levelOrder(root *TreeNode) (ans [][]int) {
	queue := make([]*TreeNode, 0)
	if root != nil {
		queue = append(queue, root)
	}
	for len(queue) > 0 {
		size := len(queue)
		level := make([]int, 0, size)
		for i := 0; i < size; i++ {
			temp := queue[0]
			queue = queue[1:]
			if temp.Left != nil {
				queue = append(queue, temp.Left)
			}
			if temp.Right != nil {
				queue = append(queue, temp.Right)
			}
			level = append(level, temp.Val)
		}
		ans = append(ans, level)
	}
	return
}

func zigzagLevelOrder(root *TreeNode) [][]int {
	level := levelOrder(root)
	for i := 0; i < len(level); i++ {
		if i%2 != 0 {
			slices.Reverse(level[i])
		}
	}
	return level
}

func maxDepth(root *TreeNode) int {
	/**
	空节点高度为0
	左子树为空 最大深度是右节点最大深度+1
	右子树为空 ...
	叶子结点高度为1
	后三种情况可以整合
	*/
	if root == nil {
		return 0
	}
	return max(maxDepth(root.Left), maxDepth(root.Right)) + 1
}

func buildTree(preorder []int, inorder []int) *TreeNode {
	if len(preorder) == 0 {
		return nil
	}
	//其实不需要每次都遍历寻找 可以用一个哈希表记录下标对应关系
	index := slices.Index(inorder, preorder[0])
	node := &TreeNode{Val: preorder[0],
		Left:  buildTree(preorder[1:index+1], inorder[:index]),
		Right: buildTree(preorder[index+1:], inorder[index+1:]),
	}
	return node
}

//func buildTree(inorder []int, postorder []int) *TreeNode {
//	if len(postorder) == 0 {
//		return nil
//	}
//	index := slices.Index(inorder, postorder[len(postorder)-1])
//	return &TreeNode{Val: postorder[len(postorder)-1],
//		Left:  buildTree(inorder[:index], postorder[:index]),
//		Right: buildTree(inorder[index+1:], postorder[index:len(postorder)-1])}
//}

func levelOrderBottom(root *TreeNode) [][]int {
	level := levelOrder(root)
	slices.Reverse(level)
	return level
}

func sortedArrayToBST(nums []int) *TreeNode {
	if len(nums) == 0 {
		return nil
	}
	middle := len(nums) / 2
	return &TreeNode{
		Val:   nums[middle],
		Left:  sortedArrayToBST(nums[:middle]),
		Right: sortedArrayToBST(nums[middle+1:]),
	}
}

// 将list转化为array
//func sortedListToBST(head *ListNode) *TreeNode {
//	nums := make([]int, 0)
//	for head != nil {
//		nums = append(nums, head.Val)
//		head = head.Next
//	}
//	return sortedArrayToBST(nums)
//}

// 每个节点的左右两个子树的高度差的绝对值不超过 1
//func isBalanced(root *TreeNode) bool {
//	if root == nil {
//		return true
//	}
//	lDepth := maxDepth(root.Left)
//	rDepth := maxDepth(root.Right)
//	return Abs(lDepth-rDepth) <= 1 && isBalanced(root.Left) && isBalanced(root.Right)
//}

//	func minDepth(root *TreeNode) int {
//		res := make([][]int, 0)
//		var dfs func(node *TreeNode, path []int)
//		dfs = func(node *TreeNode, path []int) {
//			if node == nil {
//				return
//			}
//			path = append(path, node.Val)
//			if node.Left == nil && node.Right == nil {
//				res = append(res, append([]int{}, path...))
//				return
//			}
//			dfs(node.Left, path)
//			dfs(node.Right, path)
//		}
//		if root == nil {
//			return 0
//		}
//		dfs(root, []int{})
//		minLen := math.MaxInt
//		for i := 0; i < len(res); i++ {
//			minLen = min(minLen, len(res[i]))
//		}
//		return minLen
//	}

func minDepth(root *TreeNode) int {
	//空节点高度为0
	if root == nil {
		return 0
	}
	////叶子结点高度为1
	//if root.Left == nil && root.Right == nil {
	//	return 1
	//}
	////左子树为空 则最小高度应该是右边的最小高度+1
	//if root.Left == nil {
	//	return minDepth(root.Right) + 1
	//}
	////同上
	//if root.Right == nil {
	//	return minDepth(root.Left) + 1
	//}

	//上面上种情况可以整合
	if root.Left == nil || root.Right == nil {
		if root.Left == nil && root.Right == nil {
			return 1
		}
		return max(minDepth(root.Left), minDepth(root.Right)) + 1
	}
	return min(minDepth(root.Left), minDepth(root.Right)) + 1
}

func hasPathSum(root *TreeNode, targetSum int) bool {
	var dfs func(node *TreeNode, sum int) bool
	dfs = func(node *TreeNode, sum int) bool {
		if node == nil {
			return false
		}
		sum += node.Val
		if node.Left == nil && node.Right == nil {
			return sum == targetSum
		}
		return dfs(node.Left, sum) || dfs(node.Right, sum)
	}
	return dfs(root, 0)
}

//	func pathSum(root *TreeNode, targetSum int) (ans [][]int) {
//		var dfs func(node *TreeNode, sum int, path []int)
//		dfs = func(node *TreeNode, sum int, path []int) {
//			if node == nil {
//				return
//			}
//			path = append(path, node.Val)
//			sum += node.Val
//			if node.Left == nil && node.Right == nil {
//				if sum == targetSum {
//					ans = append(ans, append([]int{}, path...))
//				}
//			}
//			dfs(node.Left, sum, path)
//			dfs(node.Right, sum, path)
//		}
//		dfs(root, 0, []int{})
//		return
//	}

func binaryTreePaths(root *TreeNode) (ans []string) {
	var dfs func(node *TreeNode, path []string)
	dfs = func(node *TreeNode, path []string) {
		if node == nil {
			return
		}
		path = append(path, strconv.Itoa(node.Val))
		if node.Left == nil && node.Right == nil {
			ans = append(ans, strings.Join(path, "->"))
			return
		}
		dfs(node.Left, path)
		dfs(node.Right, path)
	}
	dfs(root, []string{})
	return
}

func findFrequentTreeSum(root *TreeNode) (ans []int) {
	hash := make(map[int]int)
	maxCount := 0
	var dfs func(node *TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		sum := node.Val + dfs(node.Left) + dfs(node.Right)
		hash[sum]++
		maxCount = max(maxCount, hash[sum])
		return sum
	}
	dfs(root)
	for k, v := range hash {
		if v == maxCount {
			ans = append(ans, k)
		}
	}
	return
}

func tree2str(root *TreeNode) string {
	builder := strings.Builder{}
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		builder.WriteString(strconv.Itoa(node.Val))
		if node.Left != nil {
			builder.WriteByte('(')
			dfs(node.Left)
			builder.WriteByte(')')
		}
		if node.Right != nil {
			if node.Left == nil {
				builder.WriteString("()")
			}
			builder.WriteByte('(')
			dfs(node.Right)
			builder.WriteByte(')')
		}
	}
	dfs(root)
	return builder.String()
}
func addOneRow(root *TreeNode, val int, depth int) *TreeNode {
	queue := make([]*TreeNode, 0)
	if root != nil {
		queue = append(queue, root)
	}
	if depth == 1 {
		newRoot := &TreeNode{Val: val}
		newRoot.Left = root
		return newRoot
	}
	for i := 2; i < depth; i++ {
		size := len(queue)
		for j := 0; j < size; j++ {
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
	for _, node := range queue {
		node.Left = &TreeNode{Val: val, Left: node.Left}
		node.Right = &TreeNode{Val: val, Right: node.Right}
	}
	return root
}

func findDuplicateSubtrees(root *TreeNode) (ans []*TreeNode) {
	type pair struct {
		node  *TreeNode
		count int
	}
	hash := make(map[string]pair)
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		str := tree2str(node)
		if _, ok := hash[str]; !ok {
			hash[str] = pair{count: 1, node: node}
		} else {
			p := hash[str]
			p.count++
			hash[str] = p
		}
		dfs(node.Left)
		dfs(node.Right)
	}
	dfs(root)
	for _, v := range hash {
		if v.count > 1 {
			ans = append(ans, v.node)
		}
	}
	return
}
func findTarget(root *TreeNode, k int) bool {
	set := make(map[int]struct{})
	var dfs func(node *TreeNode) bool
	dfs = func(node *TreeNode) bool {
		if node == nil {
			return false
		}
		if _, ok := set[k-node.Val]; ok {
			return true
		} else {
			set[node.Val] = struct{}{}
		}
		return dfs(node.Left) || dfs(node.Right)
	}
	return dfs(root)
}
func rangeSumBST(root *TreeNode, low int, high int) (ans int) {
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		if node.Val <= high && node.Val >= low {
			ans += node.Val
		}
		dfs(node.Left)
		dfs(node.Right)
	}
	dfs(root)
	return
}
func maxLevelSum(root *TreeNode) (maxRow int) {
	maxLevel := math.MinInt
	queue := make([]*TreeNode, 0)
	if root != nil {
		queue = append(queue, root)
	}
	for row := 1; len(queue) > 0; row++ {
		size := len(queue)
		sum := 0
		for i := 0; i < size; i++ {
			temp := queue[0]
			queue = queue[1:]
			sum += temp.Val
			if temp.Left != nil {
				queue = append(queue, temp.Left)
			}
			if temp.Right != nil {
				queue = append(queue, temp.Right)
			}
		}
		if sum > maxLevel {
			maxLevel = sum
			maxRow = row
		}
	}
	return
}

//func getAllElements(root1 *TreeNode, root2 *TreeNode) (ans []int) {
//	ans = append(inorderTraversal(root1), inorderTraversal(root2)...)
//	slices.Sort(ans)
//	return
//}

//func deepestLeavesSum(root *TreeNode) (ans int) {
//	level := levelOrder(root)
//	for _, val := range level[len(level)-1] {
//		ans += val
//	}
//	return
//}

func deepestLeavesSum(root *TreeNode) int {
	hash := make(map[int]int)
	maxDep := 0
	var dfs func(node *TreeNode, depth int)
	dfs = func(node *TreeNode, depth int) {
		if node == nil {
			return
		}
		maxDep = max(maxDep, depth)
		hash[depth] += node.Val
		dfs(node.Left, depth+1)
		dfs(node.Right, depth+1)
	}
	dfs(root, 0)
	return hash[maxDep]
}
func findRepeatDocument(documents []int) int {
	set := make(map[int]struct{})
	for _, document := range documents {
		if _, ok := set[document]; !ok {
			set[document] = struct{}{}
		} else {
			return document
		}
	}
	return -1
}
func searchMatrix(matrix [][]int, target int) bool {
	for _, row := range matrix {
		if target >= row[0] && target <= row[len(row)-1] {
			_, ok := slices.BinarySearch(row, target)
			if ok {
				return true
			}
		}
	}
	return false
}

func combinationSum1(candidates []int, target int) (ans [][]int) {
	var dfs func(i, sum int, path []int)
	dfs = func(i, sum int, path []int) {
		if sum > target {
			return
		}
		if i == len(candidates) {
			if target == sum {
				ans = append(ans, append([]int{}, path...))
			}
			return
		}
		//选当前的数还可以重复再选
		sum += candidates[i]
		path = append(path, candidates[i])
		dfs(i, sum, path)
		sum -= candidates[i]
		path = path[:len(path)-1]
		//不选当前的数
		dfs(i+1, sum, path)
	}
	dfs(0, 0, []int{})
	return
}

func combinationSum2(candidates []int, target int) (ans [][]int) {
	var dfs func(i, sum int, path []int)
	dfs = func(i, sum int, path []int) {
		if sum >= target {
			if target == sum {
				ans = append(ans, append([]int{}, path...))
			}
			return
		}
		for start := i; start < len(candidates); start++ {
			path = append(path, candidates[start])
			dfs(start, sum+candidates[start], path)
			path = path[:len(path)-1]
		}
	}
	dfs(0, 0, []int{})
	return
}
func combine(n int, k int) (ans [][]int) {
	var dfs func(start int, path []int)
	dfs = func(start int, path []int) {
		if len(path) == k {
			ans = append(ans, append([]int{}, path...))
			return
		}
		for i := start; i <= n; i++ {
			path = append(path, i)
			dfs(i+1, path)
			path = path[:len(path)-1]
		}
	}
	dfs(1, []int{})
	return
}

func combinationSum3(k int, n int) (ans [][]int) {
	var dfs func(start, sum int, path []int)
	dfs = func(start, sum int, path []int) {
		if len(path) == k {
			if sum == n {
				ans = append(ans, append([]int{}, path...))
				return
			}
		}
		for i := start; i <= 9; i++ {
			sum += i
			path = append(path, i)
			dfs(i+1, sum, path)
			sum -= i
			path = path[:len(path)-1]
		}
	}
	dfs(1, 0, []int{})
	return
}

func letterCombinations(digits string) (ans []string) {
	mapping := []string{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"}
	var dfs func(index int, path []byte)
	dfs = func(index int, path []byte) {
		if len(digits) == 0 {
			return
		}
		if index == len(digits) {
			ans = append(ans, string(path))
			return
		}
		target := digits[index] - '0'
		for _, char := range mapping[target] {
			path = append(path, byte(char))
			dfs(index+1, path)
			path = path[:len(path)-1]
		}
	}
	dfs(0, []byte{})
	return
}

//func rob(nums []int) int {
//	n := len(nums)
//	if n == 1 {
//		return nums[0]
//	}
//	if n == 2 {
//		return max(nums[0], nums[1])
//	}
//	dp := 0
//	dp1, dp2 := nums[0], max(nums[0], nums[1])
//	for i := 2; i < len(nums); i++ {
//		dp = max(dp2, dp1+nums[i])
//		dp1, dp2 = dp2, dp
//	}
//	return dp
//}

/*
*
假设n=3
左子树取0个节点 右子树取2个节点 对应于dp[0]*dp[2]
左子树取1个节点 右子树取1个节点
左子树取2个节点 右子树取0个节点
...
*/
func numTrees(n int) int {
	dp := make([]int, n+1)
	dp[0] = 1
	dp[1] = 1
	for i := 2; i <= n; i++ {
		for j := 0; j < i; j++ {
			dp[i] += dp[j] * dp[i-j-1]
		}
	}
	return dp[n]
}

// 回溯法 暴力求解
//func findTargetSumWays(nums []int, target int) (ans int) {
//	var dfs func(index, sum int)
//	dfs = func(index, sum int) {
//		if index == len(nums) {
//			if sum == target {
//				ans++
//			}
//			return
//		}
//		dfs(index+1, sum+nums[index])
//		dfs(index+1, sum-nums[index])
//	}
//	dfs(0, 0)
//	return
//}

func findDuplicate(nums []int) int {
	/**
	当下标值和当前元素值不等 交换
	若要交换的位置的元素值和当前位置元素值相等 返回
	*/
	for i := 0; i < len(nums); i++ {
		for nums[i] != i+1 {
			if nums[i] == nums[nums[i]-1] {
				return nums[i]
			}
			nums[i], nums[nums[i]-1] = nums[nums[i]-1], nums[i]
		}
	}
	return -1
}
func countArrangement(n int) (ans int) {
	isUsed := make([]bool, n+1)
	var dfs func(start int)
	dfs = func(start int) {
		if start == n+1 {
			ans++
			return
		}
		for i := 1; i <= n; i++ {
			if !isUsed[i] && (i%start == 0 || start%i == 0) {
				isUsed[i] = true
				dfs(start + 1)
				isUsed[i] = false
			}
		}
	}
	dfs(1)
	return
}
func shoppingOffers(price []int, special [][]int, needs []int) int {
	ans := math.MaxInt
	var dfs func(i, curPrice int)
	//sp:当前的大礼包号
	dfs = func(sp, curPrice int) {
		//如果当前总花费已经超过了目前的最小花费 直接返回
		if curPrice > ans {
			return
		}
		//不能购买超出需求的物品
		for _, need := range needs {
			if need < 0 {
				return
			}
		}
		//大礼包购买完毕(可能一个都没有购买或者某几个礼包重复购买)
		if sp == len(special) {
			//加上购买大礼包剩下的物品总价
			for i := 0; i < len(needs); i++ {
				curPrice += needs[i] * price[i]
			}
			//取更小的
			ans = min(ans, curPrice)
			return
		}
		//购买大礼包
		for i := 0; i < len(price); i++ {
			needs[i] -= special[sp][i]
		}
		dfs(sp, curPrice+special[sp][len(price)])

		//恢复现场
		for i := 0; i < len(price); i++ {
			needs[i] += special[sp][i]
		}
		//不购买大礼包
		dfs(sp+1, curPrice)
	}
	dfs(0, 0)
	return ans
}
func maxStrength(nums []int) int64 {
	var ans int64 = math.MinInt64
	var dfs func(i int, cnt int, strength int64)
	dfs = func(i int, cnt int, strength int64) {
		if i == len(nums) {
			if strength > ans && cnt != 0 {
				ans = strength
			}
			return
		}
		dfs(i+1, cnt, strength)
		dfs(i+1, cnt+1, strength*int64(nums[i]))
	}
	dfs(0, 0, 1)
	return ans
}

// 完全二叉树编号性质：根节点序号*2为左孩子序号 再+1为右孩子序号
func widthOfBinaryTree(root *TreeNode) int {
	type Pair struct {
		Index int
		Node  *TreeNode
	}
	ans := math.MinInt
	queue := make([]Pair, 0, 3000)
	if root != nil {
		queue = append(queue, Pair{Index: 1, Node: root})
	}
	for len(queue) > 0 {
		size := len(queue)
		ans = max(ans, queue[size-1].Index-queue[0].Index+1)
		for i := 0; i < size; i++ {
			temp := queue[0]
			queue = queue[1:]
			if temp.Node.Left != nil {
				queue = append(queue, Pair{Index: 2 * temp.Index, Node: temp.Node.Left})
			}
			if temp.Node.Right != nil {
				queue = append(queue, Pair{Index: 2*temp.Index + 1, Node: temp.Node.Right})
			}
		}
	}
	return ans
}

// 前序序列化
//func serialize(root *TreeNode) string {
//	sb := strings.Builder{}
//	var dfs func(node *TreeNode)
//	dfs = func(node *TreeNode) {
//		if node == nil {
//			sb.WriteString("#,")
//			return
//		}
//		sb.WriteString(strconv.Itoa(node.Val) + ",")
//		dfs(node.Left)
//		dfs(node.Right)
//	}
//	dfs(root)
//	return sb.String()
//}

//func deserialize(serialization string) *TreeNode {
//	splits := strings.Split(serialization, ",")
//	index := 0
//	var dfs func() *TreeNode
//	dfs = func() *TreeNode {
//		if index >= len(splits) || splits[index] == "#" {
//			index++
//			return nil
//		} else {
//			val, _ := strconv.Atoi(splits[index])
//			index++
//			return &TreeNode{Val: val, Left: dfs(), Right: dfs()}
//		}
//	}
//	return dfs()
//}

// 层序序列化
// 不需要再按层处理 直接遇到一个节点就序列化一个节点 但是空节点也需要入队列
func serialize(root *TreeNode) string {
	queue := []*TreeNode{root}
	res := make([]string, 0)
	for len(queue) > 0 {
		temp := queue[0]
		queue = queue[1:]
		if temp != nil {
			res = append(res, strconv.Itoa(temp.Val))
			queue = append(queue, temp.Left, temp.Right)
		} else {
			res = append(res, "#")
		}
	}
	return strings.Join(res, ",")
}

func deserialize(serialization string) *TreeNode {
	if serialization == "#" {
		return nil
	}
	splits := strings.Split(serialization, ",")
	rootVal, _ := strconv.Atoi(splits[0])
	root := &TreeNode{Val: rootVal}
	queue := []*TreeNode{root}
	index := 1
	for index < len(splits) {
		node := queue[0]
		queue = queue[1:]
		if splits[index] != "#" {
			nodeVal, _ := strconv.Atoi(splits[index])
			node.Left = &TreeNode{Val: nodeVal}
			queue = append(queue, node.Left)
		}
		index++
		if splits[index] != "#" {
			nodeVal, _ := strconv.Atoi(splits[index])
			node.Right = &TreeNode{Val: nodeVal}
			queue = append(queue, node.Right)
		}
		index++
	}
	return root
}

// Codec 序列化和反序列化
type Codec struct {
}

//func Constructor() Codec {
//	return Codec{}
//}

func (this *Codec) serialize(root *TreeNode) string {
	return serialize(root)
}

func (this *Codec) deserialize(data string) *TreeNode {
	return deserialize(data)
}

// 出现空节点之后再次出现非空节点即非完全二叉树
// 无需按层
func isCompleteTree(root *TreeNode) bool {
	queue := []*TreeNode{root}
	hasNil := false
	for len(queue) > 0 {
		temp := queue[0]
		queue = queue[1:]
		if temp.Left != nil {
			if hasNil {
				return false
			}
			queue = append(queue, temp.Left)
		} else {
			hasNil = true
		}
		if temp.Right != nil {
			if hasNil {
				return false
			}
			queue = append(queue, temp.Right)
		} else {
			hasNil = true
		}
	}
	return true
}

// 完全二叉树一直向左遍历和一直向右遍历 深度若相同则为满二叉树
func isFullBST(root *TreeNode) (depth int, ok bool) {
	var depthLeft, depthRight int
	left, right := root, root
	for left != nil {
		depthLeft++
		left = left.Left
	}
	for right != nil {
		depthRight++
		right = right.Right
	}
	if depthLeft == depthRight {
		return depthLeft, true
	}
	return
}
func countNodes(root *TreeNode) int {
	if root == nil {
		return 0
	}
	if depth, ok := isFullBST(root); ok {
		return 1<<depth - 1
	} else {
		return countNodes(root.Left) + countNodes(root.Right) + 1
	}
}

// 不需要删除头结点 无需dummy节点
//func deleteDuplicates(head *ListNode) *ListNode {
//	if head == nil || head.Next == nil {
//		return head
//	}
//	cur := head
//	for cur.Next != nil {
//		if cur.Next.Val == cur.Val {
//			cur.Next = cur.Next.Next
//		} else {
//			cur = cur.Next
//		}
//	}
//	return head
//}

//func deleteDuplicates(head *ListNode) *ListNode {
//	dummy := &ListNode{Next: head}
//	cur := dummy
//	for cur.Next != nil && cur.Next.Next != nil {
//		val := cur.Next.Val
//		if cur.Next.Next.Val == val {
//			for cur.Next != nil && cur.Next.Val == val {
//				cur.Next = cur.Next.Next
//			}
//		} else {
//			cur = cur.Next
//		}
//	}
//	return dummy.Next
//}

// 有重复节点需要把重复的全部删除
func deleteDuplicates(head *ListNode) *ListNode {
	dummy := &ListNode{Next: head}
	cur := dummy
	for cur.Next != nil && cur.Next.Next != nil {
		val := cur.Next.Val
		if cur.Next.Next.Val == val {
			for cur.Next != nil && cur.Next.Val == val {
				cur.Next = cur.Next.Next
			}
		} else {
			cur = cur.Next
		}
	}
	return dummy.Next
}

// 偷天换日法 但不能是末尾节点
func deleteNode(node *ListNode) {
	node.Val = node.Next.Val
	node.Next = node.Next.Next
}

// 可能会删除头结点的情况下创建dummy节点
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dummy := &ListNode{Next: head}
	slow, fast := dummy, dummy
	for i := 0; i < n; i++ {
		fast = fast.Next
	}
	for fast.Next != nil {
		slow, fast = slow.Next, fast.Next
	}
	slow.Next = slow.Next.Next
	return dummy.Next
}

// 需要删除头结点 因此需要dummy
func removeElements(head *ListNode, val int) *ListNode {
	dummy := &ListNode{Next: head}
	cur := dummy
	for cur.Next != nil {
		if cur.Next.Val == val {
			cur.Next = cur.Next.Next
		} else {
			cur = cur.Next
		}
	}
	return dummy.Next
}
