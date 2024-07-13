package main

import (
	"slices"
	"strconv"
	"strings"
)

//func buildTree(inorder []int, postorder []int) *TreeNode {
//	if len(inorder) == 0 {
//		return nil
//	}
//	root := &TreeNode{Left: nil, Right: nil, Val: postorder[len(postorder)-1]}
//	index := slices.Index(inorder, root.Val)
//	root.Left = buildTree(inorder[:index], postorder[:index])
//	root.Right = buildTree(inorder[index+1:], postorder[index:len(postorder)-1])
//	return root
//}

//func getMaxIndex(nums []int) int {
//	maxIndex := 0
//	for i, num := range nums {
//		if num > nums[maxIndex] {
//			maxIndex = i
//		}
//	}
//	return maxIndex
//}
//func constructMaximumBinaryTree(nums []int) *TreeNode {
//	if len(nums) == 0 {
//		return nil
//	}
//	maxIndex := getMaxIndex(nums)
//	root := &TreeNode{Val: nums[maxIndex], Left: nil, Right: nil}
//	root.Left = constructMaximumBinaryTree(nums[:maxIndex])
//	root.Right = constructMaximumBinaryTree(nums[maxIndex+1:])
//	return root
//}
//
//// 递归法
//func mergeTrees(root1 *TreeNode, root2 *TreeNode) *TreeNode {
//	if root1 == nil || root2 == nil {
//		if root1 != nil {
//			return root1
//		}
//		return root2
//	}
//	root1.Val += root2.Val
//	root1.Left = mergeTrees(root1.Left, root2.Left)
//	root1.Right = mergeTrees(root1.Right, root2.Right)
//	return root1
//}
//
//func searchBST(root *TreeNode, val int) *TreeNode {
//	if root == nil {
//		return nil
//	}
//	if root.Val == val {
//		return root
//	} else if root.Val < val {
//		return searchBST(root.Right, val)
//	} else {
//		return searchBST(root.Left, val)
//	}
//}

// 中序遍历二叉搜索树为递增：先得到升序序列再循环判断
//func getMinimumDifference(root *TreeNode) int {
//	ans := make([]int, 0)
//	var dfs func(node *TreeNode)
//	dfs = func(node *TreeNode) {
//		if node == nil {
//			return
//		}
//		dfs(node.Left)
//		ans = append(ans, node.Val)
//		dfs(node.Right)
//	}
//	dfs(root)
//	minDiff := math.MaxInt
//	for i := 0; i < len(ans)-1; i++ {
//		minDiff = min(minDiff, ans[i+1]-ans[i])
//	}
//	return minDiff
//}

// 中序遍历二叉搜索树为递增：直接在递归过程中维护上一个值 但好像也需要对整棵树遍历 空间复杂度稍低一些
//func getMinimumDifference(root *TreeNode) int {
//	prev := -1
//	minDiff := math.MaxInt
//	var dfs func(node *TreeNode)
//	dfs = func(node *TreeNode) {
//		if node == nil {
//			return
//		}
//		dfs(node.Left)
//		if prev != -1 {
//			minDiff = min(minDiff, node.Val-prev)
//		}
//		prev = node.Val
//		dfs(node.Right)
//	}
//	dfs(root)
//	return minDiff
//}

// 丑陋无比
//func findMode(root *TreeNode) (ans []int) {
//	var maxCount, count int
//	prev := math.MaxInt
//	var dfs func(node *TreeNode)
//	dfs = func(node *TreeNode) {
//		if node == nil {
//			return
//		}
//		dfs(node.Left)
//		if prev == math.MaxInt {
//			prev = node.Val
//			count++
//			maxCount = 1
//			ans = append(ans, node.Val)
//		} else if node.Val == prev {
//			count++
//			if count > maxCount {
//				maxCount = count
//				ans = ans[len(ans)-1:]
//			} else if count == maxCount {
//				ans = append(ans, node.Val)
//			}
//		} else {
//			count = 1
//			if count == maxCount {
//				ans = append(ans, node.Val)
//			}
//			prev = node.Val
//		}
//		dfs(node.Right)
//	}
//	dfs(root)
//	return
//}

// 哈希计数实现、并时刻维护最大出现次数
// 空间复杂度较差但代码整体比较美观
//func findMode(root *TreeNode) (ans []int) {
//	maxCount := 0
//	count := make(map[int]int)
//	var dfs func(node *TreeNode)
//	dfs = func(node *TreeNode) {
//		if node == nil {
//			return
//		}
//		dfs(node.Left)
//		count[node.Val]++
//		maxCount = max(maxCount, count[node.Val])
//		dfs(node.Right)
//	}
//	dfs(root)
//	for k, v := range count {
//		if v == maxCount {
//			ans = append(ans, k)
//		}
//	}
//	return
//}

// 暴力超时
//
//	func numberOfBoomerangs(points [][]int) int {
//		distHash := make(map[[2]int]int)
//		cnt := 0
//		for i := 0; i < len(points); i++ {
//			for j := 0; j < len(points); j++ {
//				for k := 0; k < len(points); k++ {
//					if i != j && i != k && j != k {
//						var distIJ, distIK int
//						if _, ok := distHash[[2]int{min(i, j), max(i, j)}]; !ok {
//							distIJ = (points[i][0]-points[j][0])*(points[i][0]-points[j][0]) + (points[i][1]-points[j][1])*(points[i][1]-points[j][1])
//							distHash[[2]int{min(i, j), max(i, j)}] = distIJ
//						} else {
//							distIJ = distHash[[2]int{min(i, j), max(i, j)}]
//						}
//						if _, ok := distHash[[2]int{min(i, k), max(i, k)}]; !ok {
//							distIK = (points[i][0]-points[k][0])*(points[i][0]-points[k][0]) + (points[i][1]-points[k][1])*(points[i][1]-points[k][1])
//							distHash[[2]int{min(i, k), max(i, k)}] = distIK
//						} else {
//							distIK = distHash[[2]int{min(i, k), max(i, k)}]
//						}
//						if distIK == distIJ {
//							cnt++
//						}
//					}
//				}
//			}
//		}
//		return cnt
//	}
func getDist(points1, points2 []int) int {
	return (points1[0]-points2[0])*(points1[0]-points2[0]) + (points1[1]-points2[1])*(points1[1]-points2[1])
}

func numberOfBoomerangs(points [][]int) int {
	distHash := make(map[int]map[int]int)
	for i := 0; i < len(points); i++ {
		for j := i + 1; j < len(points); j++ {
			dist := getDist(points[i], points[j])
			if _, ok := distHash[i]; !ok {
				distHash[i] = make(map[int]int)
			}
			if _, ok := distHash[j]; !ok {
				distHash[j] = make(map[int]int)
			}
			distHash[i][dist]++
			distHash[j][dist]++
		}
	}
	cnt := 0
	for _, m := range distHash {
		for _, v := range m {
			cnt += v * (v - 1)
		}
	}
	return cnt
}

//	func buildTree(inorder []int, postorder []int) *TreeNode {
//		if len(inorder) == 0 {
//			return nil
//		}
//		root := &TreeNode{Left: nil, Right: nil, Val: postorder[len(postorder)-1]}
//		index := slices.Index(inorder, root.Val)
//		root.Left = buildTree(inorder[:index], postorder[:index])
//		root.Right = buildTree(inorder[index+1:], postorder[index:len(postorder)-1])
//		return root
//	}
//func buildTree(preorder []int, inorder []int) *TreeNode {
//	if len(inorder) == 0 {
//		return nil
//	}
//	root := &TreeNode{Val: preorder[0], Left: nil, Right: nil}
//	index := slices.Index(inorder, root.Val)
//	root.Left = buildTree(preorder[1:index+1], inorder[:index])
//	root.Right = buildTree(preorder[index+1:], inorder[index+1:])
//	return root
//}

func flatten(root *TreeNode) {
	var prev *TreeNode
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Right)
		dfs(node.Left)
		if prev != nil {
			node.Left = nil
			node.Right = prev
		}
		prev = node
	}
	dfs(root)
}

//func sumNumbers(root *TreeNode) int {
//	ans := make([]string, 0)
//	var dfs func(node *TreeNode, path string)
//	dfs = func(node *TreeNode, path string) {
//		if node == nil {
//			return
//		}
//		path += strconv.Itoa(node.Val)
//		if node.Left == nil && node.Right == nil {
//			ans = append(ans, path)
//		}
//		dfs(node.Left, path)
//		dfs(node.Right, path)
//	}
//	dfs(root, "")
//	res := 0
//	for _, a := range ans {
//		val, _ := strconv.Atoi(a)
//		res += val
//	}
//	return res
//}

//func sumNumbers(root *TreeNode) int {
//	ans := 0
//	var dfs func(node *TreeNode, cur int)
//	dfs = func(node *TreeNode, cur int) {
//		if node == nil {
//			return
//		}
//		cur = 10*cur + node.Val
//		if node.Left == nil && node.Right == nil {
//			ans += cur
//		}
//		dfs(node.Left, cur)
//		dfs(node.Right, cur)
//	}
//	dfs(root, 0)
//	return ans
//}

// 完整遍历得到升序序列法
//func kthSmallest(root *TreeNode, k int) int {
//	ans := make([]int, 0)
//	var dfs func(node *TreeNode)
//	dfs = func(node *TreeNode) {
//		if node == nil {
//			return
//		}
//		dfs(node.Left)
//		ans = append(ans, node.Val)
//		dfs(node.Right)
//	}
//	dfs(root)
//	return ans[k-1]
//}

// 堆排序法：大顶堆
//
//	func kthSmallest(root *TreeNode, k int) int {
//		hp := &IntHeap{}
//		var dfs func(node *TreeNode)
//		dfs = func(node *TreeNode) {
//			if node == nil {
//				return
//			}
//			dfs(node.Left)
//			heap.Push(hp, node.Val)
//			if hp.Len() > k {
//				heap.Pop(hp)
//			}
//			dfs(node.Right)
//		}
//		dfs(root)
//		fmt.Println(*hp)
//		return (*hp)[0]
//	}
//func sortedArrayToBST(nums []int) *TreeNode {
//	if len(nums) == 0 {
//		return nil
//	}
//	middle := len(nums) / 2
//	root := &TreeNode{Val: nums[middle], Left: nil, Right: nil}
//	root.Left = sortedArrayToBST(nums[:middle])
//	root.Right = sortedArrayToBST(nums[middle+1:])
//	return root
//}

//比较笨的办法
//func diameterOfBinaryTree(root *TreeNode) int {
//	maxDep := math.MinInt
//	var dfs func(node *TreeNode)
//	dfs = func(node *TreeNode) {
//		if node == nil {
//			return
//		}
//		maxDep = max(maxDep, maxDepth(node.Left)+maxDepth(node.Right))
//		dfs(node.Left)
//		dfs(node.Right)
//	}
//	dfs(root)
//	return maxDep
//}

// 回溯 返回当前节点的最大深度给上一层 上一层得到两侧的最大深度相加求解
// 这个居然就是传说中的树型DP
//func diameterOfBinaryTree(root *TreeNode) int {
//	maxDep := 0
//	var dfs func(node *TreeNode) int
//	dfs = func(node *TreeNode) int {
//		if node == nil {
//			return 0
//		}
//		leftDepth := dfs(node.Left)
//		rightDepth := dfs(node.Right)
//		maxDep = max(maxDep, leftDepth+rightDepth)
//		return max(leftDepth, rightDepth) + 1
//	}
//	dfs(root)
//	return maxDep
//}
//
//func pathSum(root *TreeNode, targetSum int) (ans [][]int) {
//	var dfs func(node *TreeNode, sum int, path []int)
//	dfs = func(node *TreeNode, sum int, path []int) {
//		if node == nil {
//			return
//		}
//		sum += node.Val
//		path = append(path, node.Val)
//		if node.Left == nil && node.Right == nil && sum == targetSum {
//			path_ := make([]int, len(path))
//			copy(path_, path)
//			ans = append(ans, path_)
//		}
//		dfs(node.Left, sum, path)
//		dfs(node.Right, sum, path)
//	}
//	dfs(root, 0, []int{})
//	return
//}

// 再写一遍二叉树的最大深度
//func maxDepth(root *TreeNode) int {
//	var dfs func(node *TreeNode) int
//	dfs = func(node *TreeNode) int {
//		if node == nil {
//			return 0
//		}
//		return max(dfs(node.Left), dfs(node.Right)) + 1
//	}
//	return dfs(root)
//}

// N叉树的最大深度
//
//	func maxDepth(root *Node) int {
//		var dfs func(node *Node) int
//		dfs = func(node *Node) int {
//			if node == nil {
//				return 0
//			}
//			res := 0
//			for i := 0; i < len(node.Children); i++ {
//				res = max(res, dfs(node.Children[i]))
//			}
//			return res + 1
//		}
//		return dfs(root)
//	}

// 双递归
//func findTilt(root *TreeNode) int {
//	ans := 0
//	var dfs func(node *TreeNode) int
//	dfs = func(node *TreeNode) int {
//		if node == nil {
//			return 0
//		}
//		if node.Left == nil && node.Right == nil {
//			return node.Val
//		}
//		leftSum := dfs(node.Left)
//		rightSum := dfs(node.Right)
//		ans += Abs(leftSum - rightSum)
//		return leftSum + rightSum + node.Val
//	}
//	dfs(root)
//	return ans
//}
//
//func treeSum(root *TreeNode) int {
//	if root == nil {
//		return 0
//	}
//	if root.Left == nil && root.Right == nil {
//		return root.Val
//	}
//	return treeSum(root.Left) + treeSum(root.Right) + root.Val
//}

// 调包侠
//func isSubtree(root *TreeNode, subRoot *TreeNode) bool {
//	var dfs func(node *TreeNode) bool
//	dfs = func(node *TreeNode) bool {
//		if node == nil {
//			return false
//		}
//		//也可以用reflect.DeepEqual()
//		if isSameTree(node, subRoot) {
//			return true
//		}
//		return dfs(node.Left) || dfs(node.Right)
//	}
//	return dfs(root)
//}

//func isSameTree(p *TreeNode, q *TreeNode) bool {
//	if p == nil || q == nil {
//		return p == q
//	} else {
//		return p.Val == q.Val && isSameTree(p.Left, q.Left) && isSameTree(p.Right, q.Right)
//	}
//}

type Node struct {
	Val      int
	Children []*Node
}

//	func preorder(root *Node) (ans []int) {
//		var dfs func(node *Node)
//		dfs = func(node *Node) {
//			if node == nil {
//				return
//			}
//			for i := 0; i < len(node.Children); i++ {
//				dfs(node.Children[i])
//			}
//			ans = append(ans, node.Val)
//		}
//		dfs(root)
//		return
//	}
//
//	func postorder(root *Node) (ans []int) {
//		var dfs func(node *Node)
//		dfs = func(node *Node) {
//			if node == nil {
//				return
//			}
//			for i := 0; i < len(node.Children); i++ {
//				dfs(node.Children[i])
//			}
//			ans = append(ans, node.Val)
//		}
//		dfs(root)
//		return
//	}
func calPoints(operations []string) int {
	pointsStack := make([]int, len(operations))
	points := 0
	for _, op := range operations {
		if op == "C" {
			points -= pointsStack[len(pointsStack)-1]
			pointsStack = pointsStack[:len(pointsStack)-1]
		} else if op == "D" {
			point := 2 * pointsStack[len(pointsStack)-1]
			points += point
			pointsStack = append(pointsStack, point)
		} else if op == "+" {
			point := pointsStack[len(pointsStack)-1] + pointsStack[len(pointsStack)-2]
			points += point
			pointsStack = append(pointsStack, point)
		} else {
			point, _ := strconv.Atoi(op)
			points += point
			pointsStack = append(pointsStack, point)
		}
	}
	return points
}

//	func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
//		l1 = reverseList(l1)
//		l2 = reverseList(l2)
//		dummy := &ListNode{Val: 0, Next: nil}
//		p := dummy
//		carry := 0
//		for l1 != nil || l2 != nil {
//			var val1, val2 int
//			p.Next = &ListNode{Val: 0, Next: nil}
//			p = p.Next
//			if l1 != nil {
//				val1 = l1.Val
//			}
//			if l2 != nil {
//				val2 = l2.Val
//			}
//			p.Val = (val1 + val2 + carry) % 10
//			carry = (val1 + val2 + carry) / 10
//			if l1 != nil {
//				l1 = l1.Next
//			}
//			if l2 != nil {
//				l2 = l2.Next
//			}
//		}
//		if carry != 0 {
//			p.Next = &ListNode{Val: 1, Next: nil}
//		}
//		return reverseList(dummy.Next)
//	}

func multiply(num1 string, num2 string) string {
	if num1 == "0" || num2 == "0" {
		return "0"
	}
	ans := ""
	for i := len(num1) - 1; i >= 0; i-- {
		builder := strings.Builder{}
		var carry byte
		for k := len(num1) - 1; k > i; k-- {
			builder.WriteByte('0')
		}
		for j := len(num2) - 1; j >= 0; j-- {
			val := (num1[i]-'0')*(num2[j]-'0') + carry
			carry = val / 10
			builder.WriteByte(val%10 + '0')
		}
		if carry > 0 {
			builder.WriteByte(carry + '0')
		}
		ans = addNums(ans, reverseString(builder.String()))
	}
	return ans
}

// 非常慢
// 思路：从树中找到到达p和q的路径 比较路径找到第一个不相等的节点 前一个就是最近公共祖先
//func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
//	var pPath, qPath []*TreeNode
//	var dfs func(node *TreeNode, path []*TreeNode)
//	dfs = func(node *TreeNode, path []*TreeNode) {
//		if node == nil || (len(pPath) != 0 && len(qPath) != 0) {
//			return
//		}
//		path = append(path, node)
//		path_ := make([]*TreeNode, len(path))
//		copy(path_, path)
//		if node.Val == p.Val {
//			pPath = path_
//		}
//		if node.Val == q.Val {
//			qPath = path_
//		}
//		dfs(node.Left, path)
//		dfs(node.Right, path)
//	}
//	dfs(root, []*TreeNode{})
//	i := 0
//	for i < len(pPath) && i < len(qPath) && pPath[i].Val == qPath[i].Val {
//		i++
//	}
//	return pPath[i-1]
//}

// 递归 分类讨论 各种情况下分别返回什么
//func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
//	var dfs func(node *TreeNode) *TreeNode
//	dfs = func(node *TreeNode) *TreeNode {
//		if node == nil || node.Val == p.Val || node.Val == q.Val {
//			return node
//		}
//		left := dfs(node.Left)
//		right := dfs(node.Right)
//		if left != nil && right != nil {
//			return node
//		}
//		if left != nil {
//			return left
//		}
//		return right
//	}
//	return dfs(root)
//}

//如果是二叉搜索树
/**
如果当前节点的值大于两个节点 则两个节点肯定都在当前节点的左边 应递归左子树找到结果
如果当前节点的值小于两个节点 则两个节点肯定都在当前节点的右边 应递归右子树找到结果
若一大一小 则两个节点一定位于当前节点的左右侧 返回当前节点
若等于其中一个 则返回当前节点
*/
//再次证明做二叉树递归的题目不要绕进具体的细节
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	if (root.Val-p.Val)*(root.Val-q.Val) <= 0 {
		return root
	}
	if root.Val > p.Val && root.Val > q.Val {
		return lowestCommonAncestor(root.Left, p, q)
	}
	return lowestCommonAncestor(root.Right, p, q)
}

/*
*
如果当前节点大于val 则说明应该在左子树中插入 若左子树为空 则插入
如果当前节点小于val 则说明应该在右子树中插入 若右子树为空 则插入
*/
//func insertIntoBST(root *TreeNode, val int) *TreeNode {
//	var dfs func(node *TreeNode)
//	dfs = func(node *TreeNode) {
//		if node.Val > val {
//			if node.Left == nil {
//				node.Left = &TreeNode{Val: val, Left: nil, Right: nil}
//				return
//			}
//			dfs(node.Left)
//		}
//		if node.Val < val {
//			if node.Right == nil {
//				node.Right = &TreeNode{Val: val, Left: nil, Right: nil}
//				return
//			}
//			dfs(node.Right)
//		}
//	}
//	if root == nil {
//		return &TreeNode{Val: val, Left: nil, Right: nil}
//	}
//	dfs(root)
//	return root
//}

// 不写额外的递归函数
func insertIntoBST(root *TreeNode, val int) *TreeNode {
	if root == nil {
		return &TreeNode{Val: val}
	}
	if root.Val > val {
		root.Left = insertIntoBST(root.Left, val)
	} else {
		root.Right = insertIntoBST(root.Right, val)
	}
	return root
}

/*
*
若当前节点大于key 则说明要删除的节点在当前节点的左边
若当前节点小于key 则说明要删除的节点在当前节点的右边
若当前节点等于key 则说明要删除的就是他

	若左右子树都为空返回空节点
	若左子树不为空 返回左子树
	若右子树不为空 返回右子树
	若左右子树都不为空 将左子树插入右子树 返回右子树
*/
//func deleteNode(root *TreeNode, key int) *TreeNode {
//	if root == nil {
//		return root
//	}
//	if root.Val > key {
//		root.Left = deleteNode(root.Left, key)
//	} else if root.Val < key {
//		root.Right = deleteNode(root.Right, key)
//	} else {
//		if root.Left != nil && root.Right != nil {
//			p := root.Right
//			for p.Left != nil {
//				p = p.Left
//			}
//			p.Left = root.Left
//			return root.Right
//		}
//		if root.Left == nil {
//			return root.Right
//		}
//		return root.Left
//	}
//	return root
//}

/*
*
如果当前节点大于high

	当前节点包括当前节点的右子树需要全部删除
	左子树中可能存在不需要删除的节点

如果当前节点小于low

	当前节点包括当前节点的左子树需要全部删除
	右子树中可能存在不需要删除的节点

如果当前节点处于范围之内 递归左右子树
*/
func trimBST(root *TreeNode, low int, high int) *TreeNode {
	if root == nil {
		return root
	}
	if root.Val > high {
		return trimBST(root.Left, low, high)
	} else if root.Val < low {
		return trimBST(root.Right, low, high)
	} else {
		root.Left = trimBST(root.Left, low, high)
		root.Right = trimBST(root.Right, low, high)
	}
	return root
}

// 后缀和法
//func convertBST(root *TreeNode) *TreeNode {
//	ans := make([]*TreeNode, 0)
//	var dfs func(node *TreeNode)
//	dfs = func(node *TreeNode) {
//		if node == nil {
//			return
//		}
//		dfs(node.Left)
//		ans = append(ans, node)
//		dfs(node.Right)
//	}
//	dfs(root)
//	sum := 0
//	for i := len(ans) - 1; i >= 0; i-- {
//		ans[i].Val += sum
//		sum = ans[i].Val
//	}
//	return root
//}

// 右中左
// 一直没转过弯来 一直想着返回值
// 其实按照后缀和的思路 一路累加即可
func convertBST(root *TreeNode) *TreeNode {
	sum := 0
	var dfs func(*TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Right)
		sum = sum + node.Val
		node.Val = sum
		dfs(node.Left)
	}
	dfs(root)
	return root
}

// Trie 字典树
//type Trie struct {
//	Children  [26]*Trie
//	EndOfWord bool
//}
//
//func Constructor() Trie {
//	return Trie{}
//}
//
//func (this *Trie) Insert(word string) {
//	p := this
//	for _, char := range word {
//		index := char - 'a'
//		if p.Children[index] == nil {
//			p.Children[index] = &Trie{}
//		}
//		p = p.Children[index]
//	}
//	p.EndOfWord = true
//}
//
//func (this *Trie) Search(word string) bool {
//	p := this
//	for _, char := range word {
//		index := char - 'a'
//		if p.Children[index] == nil {
//			return false
//		}
//		p = p.Children[index]
//	}
//	return p.EndOfWord
//}
//
//func (this *Trie) StartsWith(prefix string) bool {
//	node := this
//	for _, char := range prefix {
//		index := char - 'a'
//		if node.Children[index] == nil {
//			return false
//		}
//		node = node.Children[index]
//	}
//	return true
//}

// 字典树解法
//
//	func longestCommonPrefix(strs []string) string {
//		trie := &Trie{}
//		trie.Insert(strs[0])
//		ans := len(strs[0])
//		for i := 1; i < len(strs); i++ {
//			j := 0
//			for j < len(strs[i]) && trie.StartsWith(strs[i][:j+1]) {
//				j++
//			}
//			ans = min(ans, j)
//		}
//		return strs[0][:ans]
//	}

// WordDictionary 字典树+DFS 非常难
type WordDictionary struct {
	Children  [26]*WordDictionary
	EndOfWord bool
}

//func Constructor() WordDictionary {
//	return WordDictionary{}
//}

func (this *WordDictionary) AddWord(word string) {
	p := this
	for _, char := range word {
		index := char - 'a'
		if p.Children[index] == nil {
			p.Children[index] = &WordDictionary{}
		}
		p = p.Children[index]
	}
	p.EndOfWord = true
}

func (this *WordDictionary) Prefix(word string) bool {
	p := this
	for _, char := range word {
		index := char - 'a'
		if p.Children[index] == nil {
			return false
		}
		p = p.Children[index]
	}
	return true
}

// Search 理解非二叉树中的递归
func (this *WordDictionary) Search(word string) bool {
	var dfs func(dictionary *WordDictionary, index int) bool
	dfs = func(dictionary *WordDictionary, index int) bool {
		if len(word) == index {
			return dictionary.EndOfWord
		}
		char := word[index]
		if char != '.' {
			return dictionary.Children[char-'a'] != nil && dfs(dictionary.Children[char-'a'], index+1)
		} else {
			for _, child := range dictionary.Children {
				//一次不满足不需要立刻返回false 需要遍历完全部或者dfs为true再返回 否则走最后的false逻辑
				if child != nil && dfs(child, index+1) {
					return true
				}
			}
		}
		return false
	}
	return dfs(this, 0)
}

// 暴力自定义排序法
func lexicalOrder(n int) []int {
	nums := make([]int, n)
	for i := 0; i < n; i++ {
		nums[i] = i + 1
	}
	slices.SortFunc(nums, func(a, b int) int {
		return strings.Compare(strconv.Itoa(a), strconv.Itoa(b))
	})
	return nums
}
