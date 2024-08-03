package common

import (
	"sort"
)

// 83. 删除排序链表中的重复元素 保留一个
func deleteDuplicates(head *ListNode) *ListNode {
	var dfs func(node *ListNode) *ListNode
	dfs = func(node *ListNode) *ListNode {
		if node == nil || node.Next == nil {
			return node
		}
		if node.Val != node.Next.Val {
			node.Next = dfs(node.Next)
			return node
		} else {
			val := node.Val
			for node != nil && node.Val == val {
				node = node.Next
			}
			return &ListNode{Val: val, Next: dfs(node)}
		}
	}
	return dfs(head)
}

// 110. 平衡二叉树 左右子树高度差不超过1
func isBalanced(root *TreeNode) bool {
	var dfs func(node *TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		var left, right int
		if left = dfs(node.Left); left == -1 {
			return left
		}
		if right = dfs(node.Right); right == -1 {
			return right
		}
		if Abs(left-right) > 1 {
			return -1
		}
		return max(left, right) + 1
	}
	return dfs(root) != -1
}

// 404. 左叶子之和
func sumOfLeftLeaves(root *TreeNode) int {
	var sum int
	var dfs func(node *TreeNode, flag bool)
	dfs = func(node *TreeNode, flag bool) {
		if node == nil {
			return
		}
		if flag && node.Left == nil && node.Right == nil {
			sum += node.Val
		}
		dfs(node.Left, true)
		dfs(node.Right, false)
	}
	dfs(root, false)
	return sum
}

// 416. 分割等和子集 01背包问题 先遍历物品 再倒序遍历背包
func canPartition(nums []int) bool {
	var sum int
	for _, num := range nums {
		sum += num
	}
	if sum%2 != 0 {
		return false
	}
	target := sum / 2
	//dpi 容量为i的背包所能装的最大价值
	dp := make([]int, target+1)
	for i := 0; i < len(nums); i++ {
		for j := target; j >= nums[i]; j-- {
			dp[j] = max(dp[j], dp[j-nums[i]]+nums[i])
		}
	}
	return dp[target] == target
}

// 839. 相似字符串组 并查集
func numSimilarGroups(strs []string) int {
	var isSimilar func(str1, str2 string) bool
	isSimilar = func(str1, str2 string) bool {
		var diff int
		for i := 0; i < len(str1); i++ {
			if str1[i] != str2[i] {
				diff++
			}
		}
		if diff == 0 || diff == 2 {
			return true
		}
		return false
	}
	n := len(strs)
	father := make([]int, n)
	for i := 0; i < n; i++ {
		father[i] = i
	}
	var find func(x int) int
	find = func(x int) int {
		if father[x] != x {
			father[x] = find(father[x])
		}
		return father[x]
	}
	var union func(x, y int)
	union = func(x, y int) {
		father[find(x)] = find(y)
	}
	sets := n
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if find(i) != find(j) && isSimilar(strs[i], strs[j]) {
				union(i, j)
				sets--
			}
		}
	}
	return sets
}

// 547. 省份数量 并查集
func findCircleNum(isConnected [][]int) int {
	n := len(isConnected)
	father := make([]int, n)
	sets := n
	for i := 0; i < n; i++ {
		father[i] = i
	}
	var find func(x int) int
	find = func(x int) int {
		if father[x] != x {
			father[x] = find(father[x])
		}
		return father[x]
	}
	var union func(x, y int)
	union = func(x, y int) {
		father[find(x)] = find(y)
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if isConnected[i][j] == 1 && find(i) != find(j) {
				union(i, j)
				sets--
			}
		}
	}
	return sets
}

// 947. 移除最多的同行或同列石头
func removeStones(stones [][]int) int {
	//记录每一行和每一列的第一块石头
	//如果遍历到的石头是一行且一列的第一块石头 不用消除
	//否则合并（随便和行合并还是列合并）
	colFirst := make(map[int]int)
	rowFirst := make(map[int]int)
	n := len(stones)
	father := make([]int, n)
	sets := n
	for i := 0; i < n; i++ {
		father[i] = i
	}
	var find func(x int) int
	find = func(x int) int {
		//如果一个节点的父结点不是自己：不是一个集合的代表节点
		if father[x] != x {
			father[x] = find(father[x])
		}
		return father[x]
	}
	var union func(x, y int)
	union = func(x, y int) {
		//如果两个节点不属于一个集合：父结点不同
		if find(x) != find(y) {
			//任意将一个节点的父结点直接挂在另外一个节点的父节点上
			father[find(x)] = find(y)
			sets--
		}
	}
	for i, stone := range stones {
		row, col := stone[0], stone[1]
		//如果是一行的第一块石头
		if _, ok := rowFirst[row]; !ok {
			rowFirst[row] = i
		} else {
			union(rowFirst[row], i)
		}
		//如果是一列的第一块石头
		if _, ok := colFirst[col]; !ok {
			colFirst[col] = i
		} else {
			union(colFirst[col], i)
		}
	}
	return n - sets
}

// 2092. 找出知晓秘密的所有专家 并查集打上标签 标记知道秘密的集合代表元素
func findAllPeople(n int, meetings [][]int, firstPerson int) (ans []int) {
	sort.Slice(meetings, func(i, j int) bool {
		return meetings[i][2] < meetings[j][2]
	})

	father := make([]int, n)
	secret := make([]bool, n)
	for i := 0; i < n; i++ {
		father[i] = i
		secret[i] = false
	}
	father[firstPerson] = 0
	secret[0] = true

	var (
		find  func(x int) int
		union func(x, y int)
	)
	find = func(x int) int {
		if father[x] != x {
			father[x] = find(father[x])
		}
		return father[x]
	}
	union = func(x, y int) {
		fx, fy := find(x), find(y)
		if fx != fy {
			father[fx] = fy
			//如果x知道秘密 即secret[fx]=true 也应该让fy知道秘密
			//如果y知道秘密 无需设置 因为现在x已经挂在了y上面
			//核心语句
			secret[fy] = secret[fy] || secret[fx]
		}
	}

	for i := 0; i < len(meetings); {
		j := i
		//一次处理同一时刻的会议
		for j+1 < len(meetings) && meetings[j+1][2] == meetings[i][2] {
			j++
		}
		//将同一时刻的专家全都加入一组
		for l := i; l <= j; l++ {
			people1, people2 := meetings[l][0], meetings[l][1]
			union(people1, people2)
		}
		//如果这些专家开完会还是不知道秘密 初始化其指向
		for l := i; l <= j; l++ {
			people1, people2 := meetings[l][0], meetings[l][1]
			if !secret[find(people1)] {
				father[people1] = people1
			}
			if !secret[find(people2)] {
				father[people2] = people2
			}
		}
		i = j + 1
	}
	for i := 0; i < n; i++ {
		if secret[find(i)] {
			ans = append(ans, i)
		}
	}
	return
}

// 2421. 好路径的数目
func numberOfGoodPaths(vals []int, edges [][]int) int {
	father := make([]int, len(vals))
	maxCnt := make([]int, len(vals))
	for i := 0; i < len(vals); i++ {
		father[i] = i
		maxCnt[i] = 1
	}

	var (
		find  func(x int) int
		union func(x, y int)
		ans   = len(vals) //单节点也是好路径
	)
	//标准并查集递归find模板
	find = func(x int) int {
		if father[x] != x {
			father[x] = find(father[x])
		}
		return father[x]
	}
	union = func(x, y int) {
		fx := find(x)
		fy := find(y)
		//让最大值更大的代表节点做代表节点
		if vals[fx] > vals[fy] { //无需更新最大值个数
			father[fy] = fx
		} else if vals[fx] < vals[fy] { //无需更新最大值个数
			father[fx] = fy
		} else { //核心逻辑
			ans += maxCnt[fx] * maxCnt[fy]
			father[fy] = fx
			maxCnt[fx] += maxCnt[fy]
		}
	}

	sort.Slice(edges, func(i, j int) bool {
		return max(vals[edges[i][0]], vals[edges[i][1]]) < max(vals[edges[j][0]], vals[edges[j][1]])
	})

	for i := 0; i < len(edges); i++ {
		union(edges[i][0], edges[i][1])
	}

	return ans
}

// 200. 岛屿数量 dfs洪水填充
func numIslands(grid [][]byte) int {
	var dfs func(x, y int)
	dfs = func(x, y int) {
		if x < 0 || x >= len(grid) || y < 0 || y >= len(grid[0]) {
			return
		}
		if grid[x][y] == '1' {
			grid[x][y] = '2'
			dfs(x-1, y)
			dfs(x, y-1)
			dfs(x+1, y)
			dfs(x, y+1)
		}
	}
	var ans int
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] == '1' {
				ans++
				dfs(i, j)
			}
		}
	}
	return ans
}

// 130. 被围绕的区域
func solve(board [][]byte) {
	//从边界出发 洪水填充到的O都不会被感染 标记其不会被感染
	//再遍历一遍 将会被感染的修改为x 不会被感染的修改为f
	var dfs func(i, j int)
	dfs = func(i, j int) {
		if i < 0 || i >= len(board) || j < 0 || j >= len(board[0]) || board[i][j] == 'X' || board[i][j] == 'F' {
			return
		}
		board[i][j] = 'F'
		dfs(i-1, j)
		dfs(i+1, j)
		dfs(i, j-1)
		dfs(i, j+1)
	}

	//从边界出发
	for i := 0; i < len(board); i++ {
		if board[i][0] == 'O' { //第一列
			dfs(i, 0)
		}
		if board[i][len(board[0])-1] == 'O' { //最后一列
			dfs(i, len(board[0])-1)
		}
	}
	for i := 0; i < len(board[0]); i++ {
		if board[0][i] == 'O' { //第一行
			dfs(0, i)
		}
		if board[len(board)-1][i] == 'O' { //最后一行
			dfs(len(board)-1, i)
		}
	}
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[0]); j++ {
			if board[i][j] == 'O' {
				board[i][j] = 'X'
			} else if board[i][j] == 'F' {
				board[i][j] = 'O'
			}
		}
	}
}

// 1971. 寻找图中是否存在路径 并查集里最简单的模板题
func validPath(n int, edges [][]int, source int, destination int) bool {
	father := make([]int, n)
	for i := 0; i < n; i++ {
		father[i] = i
	}
	var (
		find  func(x int) int
		union func(x, y int)
		same  func(x, y int) bool
	)
	find = func(x int) int {
		if father[x] != x {
			father[x] = find(father[x])
		}
		return father[x]
	}
	union = func(x, y int) {
		fx := find(x)
		fy := find(y)
		if fx != fy {
			father[fx] = fy
		}
	}
	same = func(x, y int) bool {
		return find(x) == find(y)
	}

	for _, edge := range edges {
		union(edge[0], edge[1])
	}

	return same(source, destination)
}

// 拓扑排序 入度+队列
func canFinish(numCourses int, prerequisites [][]int) bool {
	graph := make([][]int, numCourses)
	inDegree := make(map[int]int)
	//建图 指向关系代表课程顺序
	for _, pre := range prerequisites {
		preCourse, course := pre[1], pre[0]
		graph[preCourse] = append(graph[preCourse], course)
		inDegree[course]++
	}
	queue := make([]int, 0, numCourses)
	//找到入度为0的节点直接加入
	for i := 0; i < numCourses; i++ {
		if inDegree[i] == 0 {
			queue = append(queue, i)
		}
	}
	var cnt int
	for len(queue) > 0 {
		temp := queue[0]
		queue = queue[1:]
		cnt++
		//找到temp的所有边
		for _, c := range graph[temp] {
			inDegree[c]--
			if inDegree[c] == 0 {
				queue = append(queue, c)
			}
		}
	}
	return cnt == numCourses
}

// 210. 课程表 II 打印拓扑排序序列 入度+队列
func findOrder(numCourses int, prerequisites [][]int) []int {
	path := make([]int, 0, numCourses)
	graph := make([][]int, numCourses)
	inDegree := make(map[int]int)
	for _, pre := range prerequisites {
		preCourse, course := pre[1], pre[0]
		graph[preCourse] = append(graph[preCourse], course)
		inDegree[course]++
	}

	queue := make([]int, 0, numCourses)
	for i := 0; i < numCourses; i++ {
		if inDegree[i] == 0 {
			queue = append(queue, i)
		}
	}
	for len(queue) > 0 {
		temp := queue[0]
		queue = queue[1:]
		path = append(path, temp)
		for _, c := range graph[temp] {
			inDegree[c]--
			if inDegree[c] == 0 {
				queue = append(queue, c)
			}
		}
	}
	if len(path) == numCourses {
		return path
	}
	return []int{}
}

// LCR 114. 火星词典 拓扑排序
//func alienOrder(words []string) string {
//	graph := make([][]byte, 26)
//	inDegree := make(map[byte]int)
//	for i := 0; i < len(words); i++ {
//		for j := i + 1; j < len(words); j++ {
//			word1 := words[i]
//			word2 := words[j]
//			//找到word1和word2第一个不相同的字符ch1 ch2
//			//加入边ch1->ch2
//			for k := 0; k < len(word1) && k < len(word2); k++ {
//				if word1[k] != word2[k] {
//					graph[word1[k]-'a'] = append(graph[word1[k]-'a'], word2[k])
//					inDegree[word2[k]]++
//					if _, ok := inDegree[word1[k]]; !ok {
//						inDegree[word1[k]] = 0
//					}
//					break
//				}
//			}
//		}
//	}
//	ans := make([]byte, 0, 26)
//	queue := make([]byte, 0, 26)
//	for i := 0; i <= 26; i++ {
//		if deg, ok := inDegree[byte(i+'a')]; ok && deg == 0 {
//			queue = append(queue)
//		}
//	}
//	for len(queue) > 0 {
//		temp := queue[1]
//		queue = queue[1:]
//		ans = append(ans, temp)
//		for _, e := range graph[temp] {
//			inDegree[e]--
//			if inDegree[e] == 0 {
//				queue = append(queue, e)
//			}
//		}
//	}
//	if len(ans) == len(inDegree) {
//		return string(ans)
//	}
//	return ""
//}

// LCR 114. 火星词典 拓扑排序
func alienOrder(words []string) string {
	graph := make([]map[int]struct{}, 26)
	for i := 0; i < len(graph); i++ {
		graph[i] = make(map[int]struct{})
	}
	inDegree := make(map[int]int)
	set := make(map[int]struct{})
	for _, word := range words {
		for _, ch := range word {
			set[int(ch-'a')] = struct{}{}
		}
	}
	for i := 0; i < len(words); i++ {
		for j := i + 1; j < len(words); j++ {
			word1 := words[i]
			word2 := words[j]
			k := 0
			for ; k < len(word1) && k < len(word2); k++ {
				if word1[k] != word2[k] {
					if _, ok := graph[word1[k]-'a'][int(word2[k]-'a')]; !ok {
						graph[word1[k]-'a'][int(word2[k]-'a')] = struct{}{}
						inDegree[int(word2[k]-'a')]++
					}
					break
				}
			}
			if (k == len(word1) || k == len(word2)) && len(word1) > len(word2) {
				return ""
			}
		}
	}
	ans := make([]byte, 0, 26)
	queue := make([]int, 0, 26)

	for k := range set {
		if inDegree[k] == 0 {
			queue = append(queue, k)
		}
	}

	for len(queue) > 0 {
		temp := queue[0]
		queue = queue[1:]
		ans = append(ans, byte(temp+'a'))
		delete(set, temp)
		for e := range graph[temp] {
			inDegree[e]--
			if inDegree[e] == 0 {
				delete(inDegree, e)
				queue = append(queue, e)
			}
		}
	}
	if len(inDegree) != 0 {
		return ""
	}
	for k := range set {
		ans = append(ans, byte(k+'a'))
	}
	return string(ans)
}
