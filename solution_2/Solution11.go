package main

import (
	"container/heap"
	"math"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"
)

func maxScoreIndices(nums []int) []int {
	scoreMap := make(map[int][]int)
	var numZero, numOne int
	for _, num := range nums {
		if num == 0 {
			numZero++
		} else {
			numOne++
		}
	}
	scoreMap[numOne] = append(scoreMap[numOne], 0)
	maxScore := max(numOne, numZero)
	scoreLeft, scoreRight := 0, numOne
	for i := 0; i < len(nums); i++ {
		if nums[i] == 0 {
			scoreLeft++
		} else if nums[i] == 1 {
			scoreRight--
		}
		if scoreLeft+scoreRight >= maxScore {
			maxScore = scoreLeft + scoreRight
			scoreMap[scoreLeft+scoreRight] = append(scoreMap[scoreLeft+scoreRight], i+1)
		}
	}
	return scoreMap[maxScore]
}

func answerQueries(nums []int, queries []int) []int {
	slices.Sort(nums)
	prefix := make([]int, len(nums))
	sum := 0
	for i, num := range nums {
		sum += num
		prefix[i] = sum
	}
	ans := make([]int, len(queries))
	for i, query := range queries {
		ans[i] = sort.SearchInts(prefix, query+1)
	}
	return ans
}
func isLetterLog(log string) bool {
	splits := strings.Split(log, " ")
	if unicode.IsLetter(rune(splits[1][0])) {
		return true
	} else {
		return false
	}
}
func reorderLogFiles(logs []string) []string {
	slices.SortStableFunc(logs, func(a, b string) int {
		if isLetterLog(a) && isLetterLog(b) {
			indexA := strings.Index(a, " ")
			indexB := strings.Index(b, " ")
			if a[indexA:] == b[indexB:] {
				return strings.Compare(a[:indexA], b[:indexB])
			} else {
				return strings.Compare(a[indexA+1:], b[indexB+1:])
			}
		} else if !isLetterLog(a) && !isLetterLog(b) {
			return 0
		} else {
			if isLetterLog(a) {
				return -1
			} else {
				return 1
			}
		}
	})
	return logs
}
func memLeak(memory1 int, memory2 int) []int {
	i := 1
	for {
		if i > memory1 && i > memory2 {
			break
		} else {
			if memory1 >= memory2 {
				memory1 -= i
			} else {
				memory2 -= i
			}
		}
		i++
	}
	return []int{i, memory1, memory2}
}

// 两次遍历实现
//
//	func swapNodes(head *ListNode, k int) *ListNode {
//		n := 0
//		for p := head; p != nil; p = p.Next {
//			n++
//		}
//		var kthNode, lastKthNode *ListNode
//		for i, p := 1, head; i <= n; i++ {
//			if i == k {
//				kthNode = p
//			}
//			if i == n-k+1 {
//				lastKthNode = p
//			}
//			p = p.Next
//		}
//		kthNode.Val, lastKthNode.Val = lastKthNode.Val, kthNode.Val
//		return head
//	}
func minimumDeletions(nums []int) int {
	maxLoc := 0
	minLoc := 0
	for i, num := range nums {
		if num > nums[maxLoc] {
			maxLoc = i
		}
		if num < nums[minLoc] {
			minLoc = i
		}
	}
	n := len(nums)
	left := min(minLoc, maxLoc)
	right := max(minLoc, maxLoc)
	return min(n-left, right+1, left+1+n-right)
}
func dayOfTheWeek(day int, month int, year int) string {
	return time.Date(year, time.Month(month), day, 0, 0, 0, 0, time.Local).Weekday().String()
}
func minFlips(a int, b int, c int) int {
	ans := 0
	for a > 0 || b > 0 || c > 0 {
		aByte, bByte, cByte := a&1, b&1, c&1
		if cByte == 0 {
			if aByte == 1 && bByte == 1 {
				ans += 2
			} else if aByte&bByte == 0 && aByte|bByte == 1 {
				ans += 1
			}
		} else if cByte == 1 {
			if aByte == 0 && bByte == 0 {
				ans += 1
			}
		}
		a, b, c = a>>1, b>>1, c>>1
	}
	return ans
}

// 58624 5=> index=3 586 5 24
func maxValue(n string, x int) string {
	index := 0
	if n[0] != '-' {
		index = strings.IndexFunc(n, func(r rune) bool {
			return r < rune(x+'0')
		})
	} else {
		index = strings.IndexFunc(n, func(r rune) bool {
			return r > rune(x+'0')
		})
	}
	if index == -1 {
		return n + strconv.Itoa(x)
	} else {
		return n[:index] + strconv.Itoa(x) + n[index:]
	}
}
func matchPlayersAndTrainers(players []int, trainers []int) int {
	slices.Sort(players)
	slices.Sort(trainers)
	ans := 0
	start := 0
	for _, player := range players {
		trainers = trainers[start:]
		index := sort.SearchInts(trainers, player)
		if index != len(trainers) {
			ans++
			start = index + 1
		} else {
			break
		}
	}
	return ans
}
func countPoints(points [][]int, queries [][]int) []int {
	ans := make([]int, len(queries))
	for i, query := range queries {
		cnt := 0
		for _, point := range points {
			if (point[0]-query[0])*(point[0]-query[0])+(point[1]-query[1])*(point[1]-query[1]) <= query[2]*query[2] {
				cnt++
			}
		}
		ans[i] = cnt
	}
	return ans
}
func executeInstructions(n int, startPos []int, s string) []int {
	instructions := []byte(s)
	ans := make([]int, len(instructions))
	position := make([]int, len(startPos))
	for i := 0; i < len(instructions); i++ {
		copy(position, startPos)
		cnt := 0
		for j := i; j < len(instructions); j++ {
			switch instructions[j] {
			case 'U':
				position[0] -= 1
			case 'D':
				position[0] += 1
			case 'L':
				position[1] -= 1
			case 'R':
				position[1] += 1
			}
			if position[0] > n-1 || position[0] < 0 || position[1] > n-1 || position[1] < 0 {
				break
			} else {
				cnt++
			}
		}
		ans[i] = cnt
	}
	return ans
}
func numPairsDivisibleBy60(time []int) int {
	count := make(map[int]int)
	for _, t := range time {
		count[t]++
	}
	ans := 0
	for _, t := range time {
		count[t]--
		//遍历太慢
		for sum := (t/60 + 1) * 60; sum <= 960; sum += 60 {
			ans += count[sum-t]
		}
	}
	return ans
}
func prefixesDivBy5(nums []int) []bool {
	ans := make([]bool, len(nums))
	n := 0
	for i := 0; i < len(nums); i++ {
		n = (n<<1 | nums[i]) % 10
		ans[i] = n%5 == 0
	}
	return ans
}
func fib(n int) int {
	dp, dp1 := 0, 1
	if n < 2 {
		return n
	} else {
		dp2 := 0
		for i := 2; i <= n; i++ {
			dp2 = dp + dp1
			dp, dp1 = dp1, dp2
		}
		return dp2
	}
}

//func climbStairs(n int) int {
//	if n <= 2 {
//		return n
//	} else {
//		dp, dp1 := 1, 2
//		dp2 := 0
//		for i := 3; i <= n; i++ {
//			dp2 = dp + dp1
//			dp, dp1 = dp1, dp2
//		}
//		return dp2
//	}
//}

// cost数组含义：从第i层向上跳的花费是cost[i]
func minCostClimbingStairs(cost []int) int {
	//dp数组含义：到达第i-1层的最小花费
	dp := make([]int, len(cost)+1)
	/**
	如何初始化:与默认初始化相同 无需显式初始化
		dp[0] = 0
		dp[1] = 0
	*/
	//遍历顺序：从前到后
	for i := 2; i <= len(cost); i++ {
		//递推公式 状态转移方程
		//想到达第i层可以从i-1层跳一步上来、也可以从i-2层跳两步上来：取二者最小值
		dp[i] = min(dp[i-1]+cost[i-1], dp[i-2]+cost[i-2])
	}
	return dp[len(dp)-1]
}

// 简单的二维dp
func uniquePaths(m int, n int) int {
	//dp数组含义：从初始位置到达i,j位置的方法数
	dp := make([][]int, 0, m)
	for i := 0; i < m; i++ {
		dp = append(dp, make([]int, n))
	}
	//初始化认为原地不动也算作一种方法
	dp[0][0] = 1
	//最上面一行需要初始化
	for i := 1; i < n; i++ {
		dp[0][i] = 1
	}
	//最左面一列需要初始化
	for i := 1; i < m; i++ {
		dp[i][0] = 1
	}
	//遍历顺序为什么是从上向下从左到右：递推公式需要上和左边的值推导
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			//递推公式、状态转移方程
			dp[i][j] = dp[i-1][j] + dp[i][j-1]
		}
	}
	return dp[m-1][n-1]
}
func uniquePathsWithObstacles(obstacleGrid [][]int) int {
	m := len(obstacleGrid)
	n := len(obstacleGrid[0])
	dp := make([][]int, 0, m)
	for i := 0; i < m; i++ {
		dp = append(dp, make([]int, n))
	}
	//第一行和第一列一旦遇到一个障碍物，后续的全都初始化为0
	for i := 0; i < n && obstacleGrid[0][i] != 1; i++ {
		dp[0][i] = 1
	}
	for i := 0; i < m && obstacleGrid[i][0] != 1; i++ {
		dp[i][0] = 1
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			//障碍物点无论从上面还是从左面来都是0
			if obstacleGrid[i][j] == 1 {
				dp[i][j] = 0
			} else {
				dp[i][j] = dp[i-1][j] + dp[i][j-1]
			}
		}
	}
	return dp[m-1][n-1]
}
func generate(numRows int) (ans [][]int) {
	for i := 0; i < numRows; i++ {
		res := make([]int, i+1)
		res[0] = 1
		res[i] = 1
		if i >= 2 {
			for j := 1; j < i; j++ {
				res[j] = ans[i-1][j-1] + ans[i-1][j]
			}
		}
		ans = append(ans, res)
	}
	return ans
}

func getRow(rowIndex int) []int {
	var prev []int
	var res []int
	for i := 0; i <= rowIndex; i++ {
		res = make([]int, i+1)
		res[0] = 1
		res[i] = 1
		if i >= 2 {
			for j := 1; j < i; j++ {
				res[j] = prev[j-1] + prev[j]
			}
		}
		prev = res
	}
	return res
}

// 内置函数解决
//
//	func countBits(n int) []int {
//		ans := make([]int, n+1)
//		for i := 0; i <= n; i++ {
//			ans[i] = bits.OnesCount(uint(i))
//		}
//		return ans
//	}

// 动态规划解决
func countBits(n int) []int {
	//dp数组含义：dp[i]为数字i的二进制表示的1的个数
	dp := make([]int, n+1)
	for i := 2; i <= n; i++ {
		//状态转移方程：去掉最后一位的1个数加上最后一位是否为1
		dp[i] = dp[i>>1]
		if i&1 == 1 {
			dp[i] += 1
		}
	}
	return dp
}

// 字符串做法
//
//	func isSubsequence(s string, t string) bool {
//		start := 0
//		for i := 0; i < len(s); i++ {
//			if offset := strings.Index(t[start:], s[i:i+1]); offset == -1 {
//				return false
//			} else {
//				start += offset + 1
//			}
//		}
//		return true
//	}

// 暴力做法
//func numWays(n int, relation [][]int, k int) int {
//	reach := make(map[int][]int)
//	for _, r := range relation {
//		reach[r[0]] = append(reach[r[0]], r[1])
//	}
//	temp := []int{0}
//	var ans []int
//	for i := 1; i <= k; i++ {
//		ans = make([]int, 0)
//		for j := 0; j < len(temp); j++ {
//			ans = append(ans, reach[temp[j]]...)
//		}
//		temp = ans
//	}
//	cnt := 0
//	for i := 0; i < len(ans); i++ {
//		if ans[i] == n-1 {
//			cnt++
//		}
//	}
//	return cnt
//}

func numWays(n int, relation [][]int, k int) int {
	dp := make([][]int, 0, k+1)
	for i := 0; i < k+1; i++ {
		dp = append(dp, make([]int, n))
	}
	dp[0][0] = 1
	for i := 1; i <= k; i++ {
		for _, r := range relation {
			dp[i][r[1]] += dp[i-1][r[0]]
		}
	}
	return dp[k][n-1]
}

// 常规做法
//
//	func maxSales(sales []int) int {
//		maxSale := math.MinInt
//		for i := 0; i < len(sales); {
//			sale := 0
//			j := i
//			for ; j < len(sales); j++ {
//				sale += sales[j]
//				maxSale = max(maxSale, sale)
//				if sale <= 0 {
//					break
//				}
//			}
//			i = j + 1
//		}
//		return maxSale
//	}
//

// 动态规划做法
func maxSales(sales []int) int {
	dp := make([]int, len(sales))
	dp[0] = sales[0]
	for i := 1; i < len(sales); i++ {
		dp[i] = max(dp[i-1]+sales[i], sales[i])
	}
	return slices.Max(dp)
}

func leastMinutes(n int) int {
	p := 0
	for math.Pow(2, float64(p)) < float64(n) {
		p++
	}
	return p + 1
}

func waysToStep(n int) int {
	if n == 1 {
		return 1
	} else if n == 2 {
		return 2
	} else if n == 3 {
		return 4
	}
	dp1, dp2, dp3 := 1, 2, 4
	dp := 0
	for i := 4; i <= n; i++ {
		dp = (dp1 + dp2 + dp3) % 1000000007
		dp1, dp2, dp3 = dp2, dp3, dp
	}
	return dp
}

func reverseBits(num int) int {
	builder := strings.Builder{}
	for i := 0; i < 32; i++ {
		builder.WriteByte(byte(num&1 + '0'))
		num >>= 1
	}
	ones := strings.Split(builder.String(), "0")
	maxLength := len(ones[0])
	for i := 1; i < len(ones); i++ {
		maxLength = max(maxLength, len(ones[i-1])+len(ones[i])+1)
	}
	return maxLength
}

func massage(nums []int) int {
	if len(nums) == 0 {
		return 0
	} else if len(nums) == 1 {
		return nums[0]
	} else if len(nums) == 2 {
		return max(nums[0], nums[1])
	} else {
		dp := make([]int, len(nums))
		dp[0] = nums[0]
		dp[1] = max(nums[0], nums[1])
		for i := 2; i < len(nums); i++ {
			dp[i] = max(dp[i-1], dp[i-2]+nums[i])
		}
		return dp[len(nums)-1]
	}
}

// 本题最难的点在于如何理解状态转移方程
func integerBreak(n int) int {
	dp := make([]int, n+1)
	dp[1] = 1
	dp[2] = 1
	for i := 3; i <= n; i++ {
		//需要依次遍历每一种可能的拆分情况
		for j := 1; j < i; j++ {
			//dp[i] = max(dp[i],...)是为了保存每一次遍历的最大值
			//j*(i-j)代表将i拆分为j和i-j两个值的情况
			//j*dp[i-j]代表之前计算过的i-j的最大拆分结果
			dp[i] = max(dp[i], max(j*(i-j), j*dp[i-j]))
		}
	}
	return dp[n]
}

// 非常基础的二维dp
func minPathSum(grid [][]int) int {
	dp := make([][]int, 0, len(grid))
	for i := 0; i < len(grid); i++ {
		dp = append(dp, make([]int, len(grid[0])))
	}
	dp[0][0] = grid[0][0]
	for i := 1; i < len(grid[0]); i++ {
		dp[0][i] = grid[0][i] + dp[0][i-1]
	}
	for i := 1; i < len(grid); i++ {
		dp[i][0] = grid[i][0] + dp[i-1][0]
	}
	for i := 1; i < len(grid); i++ {
		for j := 1; j < len(grid[0]); j++ {
			dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
		}
	}
	return dp[len(grid)-1][len(grid[0])-1]
}
func minimumTotal(triangle [][]int) int {
	dp := make([][]int, 0, len(triangle))
	for i := 1; i <= len(triangle); i++ {
		dp = append(dp, make([]int, i))
	}
	dp[0][0] = triangle[0][0]
	for i := 1; i < len(triangle); i++ {
		for j := 0; j < i+1; j++ {
			if j == 0 {
				dp[i][j] = dp[i-1][j] + triangle[i][j]
			} else if j == i {
				dp[i][j] = dp[i-1][j-1] + triangle[i][j]
			} else {
				dp[i][j] = min(dp[i-1][j-1], dp[i-1][j]) + triangle[i][j]
			}
		}
	}
	return slices.Min(dp[len(triangle)-1])
}
func rotatedDigits(n int) int {
	dp := []int{0, 0, 1, -1, -1, 1, 1, -1, 0, 1}
	cnt := 0
	for i := 0; i <= n; i++ {
		if i >= 10 {
			a := i / 10
			b := i % 10
			if dp[a] == -1 {
				dp = append(dp, -1)
			} else if dp[a] == 0 {
				dp = append(dp, dp[b])
			} else {
				if dp[b] >= 0 {
					dp = append(dp, 1)
				} else {
					dp = append(dp, -1)
				}
			}
		}
		if dp[i] == 1 {
			cnt++
		}
	}
	return cnt
}
func longestSubarray(nums []int) int {
	ones := make([][]int, 0, len(nums))
	start := 0
	for i := 0; i < len(nums); i++ {
		if nums[i] == 0 {
			ones = append(ones, nums[start:i])
			start = i + 1
		}
		if i == len(nums)-1 {
			ones = append(ones, nums[start:])
		}
	}
	if len(ones) == 1 {
		return len(ones[0]) - 1
	}
	ans := math.MinInt
	for i := 0; i < len(ones)-1; i++ {
		ans = max(ans, len(ones[i])+len(ones[i+1]))
	}
	return ans
}
func getWeight(n int) int {
	cnt := 0
	for n != 1 {
		cnt++
		if n%2 == 0 {
			n /= 2
		} else {
			n = n*3 + 1
		}
	}
	return cnt
}

//type PairHeap [][2]int
//
//func (p PairHeap) Len() int {
//	return len(p)
//}
//
//func (p PairHeap) Less(i, j int) bool {
//	if p[i][0] == p[j][0] {
//		return p[i][1] < p[j][1]
//	} else {
//		return p[i][0] < p[j][0]
//	}
//}
//
//func (p PairHeap) Swap(i, j int) {
//	p[i], p[j] = p[j], p[i]
//}
//
//func (p *PairHeap) Push(x any) {
//	*p = append(*p, x.([2]int))
//}
//
//func (p *PairHeap) Pop() any {
//	x := (*p)[len(*p)-1]
//	*p = (*p)[:len(*p)-1]
//	return x
//}

func getKth(lo int, hi int, k int) int {
	hp := &PairHeap{}
	for i := lo; i <= hi; i++ {
		heap.Push(hp, [2]int{getWeight(i), i})
		if hp.Len() > k {
			heap.Pop(hp)
		}
	}
	return heap.Pop(hp).([2]int)[1]
}
func numSplits(s string) int {
	left := make(map[byte]int)
	right := make(map[byte]int)
	for _, char := range s {
		right[byte(char)]++
	}
	ans := 0
	for _, char := range s {
		left[byte(char)]++
		right[byte(char)]--
		if right[byte(char)] == 0 {
			delete(right, byte(char))
		}
		if len(left) == len(right) {
			ans++
		}
	}
	return ans
}
func missingNumber(nums []int) int {
	sum := 0
	n := len(nums)
	for _, num := range nums {
		sum += num
	}
	return n*(n+1)/2 - sum
}

//func traversal(node *TreeNode, ans *[]int) {
//	if node == nil {
//		return
//	} else {
//		*ans = append(*ans, node.Val)
//		traversal(node.Left, ans)
//		traversal(node.Right, ans)
//	}
//}

// 递归
func preorderTraversal(root *TreeNode) (ans []int) {
	var traversal func(*TreeNode)
	traversal = func(node *TreeNode) {
		if node == nil {
			return
		} else {
			ans = append(ans, node.Val)
			traversal(node.Left)
			traversal(node.Right)
		}
	}
	traversal(root)
	return
}

// 非递归 迭代法
//func preorderTraversal(root *TreeNode) (ans []int) {
//	if root == nil {
//		return
//	}
//	stack := make([]*TreeNode, 0)
//	stack = append(stack, root)
//	for len(stack) > 0 {
//		top := stack[len(stack)-1]
//		ans = append(ans, top.Val)
//		stack = stack[:len(stack)-1]
//		if top.Right != nil {
//			stack = append(stack, top.Right)
//		}
//		if top.Left != nil {
//			stack = append(stack, top.Left)
//		}
//	}
//	return
//}

// 递归
//	func postorderTraversal(root *TreeNode) (ans []int) {
//		var traversal func(*TreeNode)
//		traversal = func(node *TreeNode) {
//			if node == nil {
//				return
//			} else {
//				traversal(node.Left)
//				traversal(node.Right)
//				ans = append(ans, node.Val)
//			}
//		}
//		traversal(root)
//		return
//	}

// 非递归 迭代法
//func postorderTraversal(root *TreeNode) (ans []int) {
//	if root == nil {
//		return
//	}
//	stack := make([]*TreeNode, 0)
//	stack = append(stack, root)
//	for len(stack) > 0 {
//		top := stack[len(stack)-1]
//		ans = append(ans, top.Val)
//		stack = stack[:len(stack)-1]
//		if top.Left != nil {
//			stack = append(stack, top.Left)
//		}
//		if top.Right != nil {
//			stack = append(stack, top.Right)
//		}
//	}
//	slices.Reverse(ans)
//	return
//}

// 递归
func inorderTraversal(root *TreeNode) (res []int) {
	var inorder func(node *TreeNode)
	inorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		inorder(node.Left)
		res = append(res, node.Val)
		inorder(node.Right)
	}
	inorder(root)
	return
}

//	func removeNodes(head *ListNode) *ListNode {
//		//单调递减栈 找到右侧第一个比自己大的
//		stack := make([]int, 0)
//		for p := head; p != nil; p = p.Next {
//			for len(stack) > 0 && stack[len(stack)-1] < p.Val {
//				stack = stack[:len(stack)-1]
//			}
//			stack = append(stack, p.Val)
//		}
//		p := head
//		for i := 0; i < len(stack); i++ {
//			p.Val = stack[i]
//			if i != len(stack)-1 {
//				p = p.Next
//			}
//		}
//		p.Next = nil
//		return head
//	}
//
// func sumOfNumberAndReverse(num int) bool {
//
// }
func reverseNum(x int) int {
	value := 0
	for ; x > 0; x = x / 10 {
		value *= 10
		value += x % 10
	}
	return value
}
func sumOfNumberAndReverse(num int) bool {
	for i := 0; i <= num; i++ {
		if i+reverseNum(i) == num {
			return true
		}
	}
	return false
}
func isPrime(num int) bool {
	if num == 1 {
		return false
	} else if num <= 3 {
		return true
	} else {
		for i := 2; i <= int(math.Sqrt(float64(num))); i++ {
			if num%i == 0 {
				return false
			}
		}
		return true
	}
}

func diagonalPrime(nums [][]int) int {
	maxPrime := 0
	for i := 0; i < len(nums); i++ {
		if isPrime(nums[i][i]) {
			maxPrime = max(maxPrime, nums[i][i])
		}
		if isPrime(nums[i][len(nums)-i-1]) {
			maxPrime = max(maxPrime, nums[i][len(nums)-i-1])
		}
	}
	return maxPrime
}
func removeCoveredIntervals(intervals [][]int) int {
	slices.SortFunc(intervals, func(a, b []int) int {
		if a[0] == b[0] {
			return b[1] - a[1]
		}
		return a[0] - b[0]
	})
	cnt := 0
	for i := 0; i < len(intervals); i++ {
		for j := 0; j < i; j++ {
			if intervals[j][1] >= intervals[i][1] {
				cnt++
				break
			}
		}
	}
	return len(intervals) - cnt
}
func countServers(grid [][]int) int {
	rowCount := make([]int, len(grid))
	colCount := make([]int, len(grid[0]))
	for i, row := range grid {
		for j, num := range row {
			if num == 1 {
				rowCount[i]++
				colCount[j]++
			}
		}
	}
	cnt := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] == 1 && (rowCount[i] > 1 || colCount[j] > 1) {
				cnt++
			}
		}
	}
	return cnt
}
func trimMean(arr []int) float64 {
	slices.Sort(arr)
	trim := len(arr) / 20
	sum := 0
	for i := trim; i < len(arr)-trim; i++ {
		sum += arr[i]
	}
	return float64(sum) / float64(len(arr)-2*trim)
}
func findMatrix(nums []int) (ans [][]int) {
	count := make(map[int]int)
	for _, num := range nums {
		count[num]++
	}
	for {
		temp := make([]int, 0, len(nums))
		for k, v := range count {
			if v > 0 {
				temp = append(temp, k)
				count[k]--
			}
		}
		if len(temp) > 0 {
			ans = append(ans, temp)
		} else {
			break
		}
	}
	return
}
func sequentialDigits(low int, high int) []int {
	seq := []int{12, 23, 34, 45, 56, 67, 78, 89, 123, 234, 345, 456, 567, 678, 789, 1234, 2345, 3456, 4567, 5678, 6789, 12345, 23456, 34567, 45678, 56789, 123456, 234567, 345678, 456789, 1234567, 2345678, 3456789, 12345678, 23456789, 123456789}
	left := sort.SearchInts(seq, low)
	right := sort.SearchInts(seq, high+1)
	return seq[left:right]
}
