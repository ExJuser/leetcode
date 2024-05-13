package main

import (
	"container/heap"
	"math"
	"slices"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

// 贪心算法分发饼干
// 每个孩子只给一块
func findContentChildren(g []int, s []int) int {
	slices.Sort(g)
	slices.Sort(s)
	var i, j int
	for ; i < len(g); i++ {
		index := sort.SearchInts(s[j:], g[i])
		if index == len(s)-j {
			return i
		} else {
			j += index + 1
		}
	}
	return i
}

// 二叉树的所有路径
func binaryTreePaths(root *TreeNode) (ans []string) {
	var dfs func(node *TreeNode, path []string)
	dfs = func(node *TreeNode, path []string) {
		if node == nil {
			return
		}
		path = append(path, strconv.Itoa(node.Val))
		if node.Left == nil && node.Right == nil {
			ans = append(ans, strings.Join(path, "->"))
		}
		dfs(node.Left, path)
		dfs(node.Right, path)
	}
	dfs(root, []string{})
	return
}

// 找出所有从根节点到叶子节点路径总和等于给定目标和的路径。
func pathSum(root *TreeNode, targetSum int) (ans [][]int) {
	var dfs func(node *TreeNode, sum int, path []int)
	dfs = func(node *TreeNode, sum int, path []int) {
		if node == nil {
			return
		}
		path = append(path, node.Val)
		sum += node.Val
		if node.Left == nil && node.Right == nil && sum == targetSum {
			ans = append(ans, append([]int{}, path...))
		}
		dfs(node.Left, sum, path)
		dfs(node.Right, sum, path)
	}
	dfs(root, 0, []int{})
	return
}

// 字母大小写全排列
func letterCasePermutation(s string) (ans []string) {
	var dfs func(index int, path []byte)
	dfs = func(index int, path []byte) {
		if index == len(s) {
			ans = append(ans, string(path))
			return
		}
		ch := rune(s[index])
		if unicode.IsLetter(ch) {
			dfs(index+1, append(path, byte(unicode.ToLower(ch))))
			dfs(index+1, append(path, byte(unicode.ToUpper(ch))))
		}
	}
	dfs(0, []byte{})
	return
}

// 烹饪料理
func perfectMenu(materials []int, cookbooks [][]int, attribute [][]int, limit int) int {
	/**
	materials[j] 表示第 j 种食材的数量
	cookbooks[i][j] 表示制作第 i 种料理需要第 j 种食材的数量
	attribute[i] = [x,y] 表示第 i 道料理的美味度 x 和饱腹感 y
	饱腹感不小于 limit 的情况下，请返回勇者可获得的最大美味度
	*/
	ans := -1
	var dfs func(menuIndex, full, yummy int)
	dfs = func(menuIndex, full, yummy int) {
		//做完了所有的料理
		if menuIndex == len(cookbooks) {
			//饱腹感达到要求
			if full >= limit {
				//更新最大美味度
				ans = max(ans, yummy)
			}
			return
		}
		flag := true
		for j := 0; j < len(cookbooks[menuIndex]); j++ {
			if materials[j] < cookbooks[menuIndex][j] {
				flag = false
				break
			}
		}
		if flag {
			for j := 0; j < len(cookbooks[menuIndex]); j++ {
				materials[j] -= cookbooks[menuIndex][j]
			}
			dfs(menuIndex+1, full+attribute[menuIndex][1], yummy+attribute[menuIndex][0])
			for j := 0; j < len(cookbooks[menuIndex]); j++ {
				materials[j] += cookbooks[menuIndex][j]
			}
		}
		dfs(menuIndex+1, full, yummy)
	}
	dfs(0, 0, 0)
	return ans
}

// 摆动序列 动态规划解法
func wiggleMaxLength(nums []int) int {
	dp1 := make([]int, len(nums))
	dp2 := make([]int, len(nums))
	for i := 0; i < len(nums); i++ {
		dp1[i] = 1
		dp2[i] = 1
	}
	for i := 1; i < len(nums); i++ {
		for j := 0; j < i; j++ {
			if dp1[j]%2 == 0 && nums[i] < nums[j] || dp1[j]%2 != 0 && nums[i] > nums[j] {
				dp1[i] = max(dp1[i], dp1[j]+1)
			}
			if dp2[j]%2 == 0 && nums[i] > nums[j] || dp2[j]%2 != 0 && nums[i] < nums[j] {
				dp2[i] = max(dp2[i], dp2[j]+1)
			}
		}
	}
	return max(slices.Max(dp1), slices.Max(dp2))
}

// 最大子数组和的最优解
func maxSubArray(nums []int) int {
	ans := nums[0]
	for i := 1; i < len(nums); i++ {
		nums[i] = max(nums[i], nums[i-1]+nums[i])
		ans = max(ans, nums[i])
	}
	return ans
}

// 买卖股票的最佳时机2 dp解法
func maxProfit(prices []int) int {
	//[0]:持有股票
	//[1]:不持有股票
	dp := make([][2]int, len(prices))
	dp[0][0] = -prices[0]
	for i := 1; i < len(prices); i++ {
		//在i天持有股票：i-1天已经持有、i天买入
		dp[i][0] = max(dp[i-1][0], dp[i-1][1]-prices[i])
		//i天不持有股票：i-1天不持有、i-1天持有i天卖出
		dp[i][1] = max(dp[i-1][1], dp[i-1][0]+prices[i])
	}
	return dp[len(prices)-1][1]
}

// 跳跃游戏
// 2,3,1,1,4
// 覆盖范围
func canJump(nums []int) bool {
	maxRight := 0
	for i := 0; i <= maxRight && i < len(nums); i++ {
		maxRight = max(maxRight, i+nums[i])
	}
	return maxRight >= len(nums)-1
}

// 跳跃游戏2
//
//	func jump(nums []int) int {
//		dp := make([]int, len(nums))
//		for i := 1; i < len(nums); i++ {
//			dp[i] = math.MaxInt
//		}
//		for i := 0; i < len(nums); i++ {
//			for j := 1; j <= nums[i] && j < len(nums)-i; j++ {
//				dp[i+j] = min(dp[i+j], dp[i]+1)
//			}
//		}
//		return dp[len(nums)-1]
//	}

func jump(nums []int) int {
	var maxRight, right, steps int
	for i := 0; i < len(nums)-1; i++ {
		maxRight = max(maxRight, i+nums[i])
		if i == right {
			right = maxRight
			steps++
		}
	}
	return steps
}

type IntHeap []int

func (h *IntHeap) Len() int {
	return len(*h)
}

func (h *IntHeap) Less(i, j int) bool {
	return (*h)[i] < (*h)[j]
}

func (h *IntHeap) Swap(i, j int) {
	(*h)[i], (*h)[j] = (*h)[j], (*h)[i]
}

func (h *IntHeap) Push(x any) {
	*h = append(*h, x.(int))
}

func (h *IntHeap) Pop() any {
	x := (*h)[h.Len()-1]
	*h = (*h)[:h.Len()-1]
	return x
}

func largestSumAfterKNegations(nums []int, k int) int {
	hp := &IntHeap{}
	*hp = nums
	heap.Init(hp)
	ans := 0
	for k > 0 {
		x := heap.Pop(hp).(int)
		heap.Push(hp, -x)
		k--
	}
	for _, num := range *hp {
		ans += num
	}
	return ans
}

func canCompleteCircuit(gas []int, cost []int) int {
	var start, leftGas, gasSum, costSum int
	for i := 0; i < len(gas); i++ {
		gasSum += gas[i]
		costSum += cost[i]
		leftGas += gas[i] - cost[i]
		//从start到i的位置都不能作为起点 直接选择i+1作为新的起点
		if leftGas < 0 {
			start = i + 1
			leftGas = 0
		}
	}
	if gasSum < costSum {
		return -1
	}
	return start
}

// 分发糖果
func candy(ratings []int) int {
	candies := make([]int, len(ratings))
	ans := 0
	for _, i := range candies {
		candies[i] = 1
	}
	for i := 0; i < len(candies); i++ {
		if i != 0 && ratings[i] > ratings[i-1] {
			candies[i] = candies[i-1] + 1
		}
	}
	for i := len(candies) - 1; i >= 0; i-- {
		if i != len(candies)-1 && ratings[i] > ratings[i+1] {
			candies[i] = candies[i+1] + 1
		}
	}
	for c := range candies {
		ans += c
	}
	return ans
}

// if else
func lemonadeChange(bills []int) bool {
	var five, ten, twenty int
	for _, bill := range bills {
		if bill == 5 {
			five++
		} else if bill == 10 {
			if five == 0 {
				return false
			} else {
				five--
				ten++
			}
		} else if bill == 20 {
			if five == 0 {
				return false
			} else {
				//优先找10+5
				if ten == 0 {
					if five < 3 {
						return false
					} else {
						five -= 3
						twenty++
					}
				} else {
					five--
					ten--
					twenty++
				}
			}
		}
	}
	return true
}

func reconstructQueue(people [][]int) [][]int {
	slices.SortFunc(people, func(a, b []int) int {
		if a[0] == b[0] {
			return a[1] - b[1]
		}
		return a[0] - b[0]
	})
	for i, p := range people {
		if p[1] < i {
			//需要将其插入到i位置
			temp := p
			for j := i; j > p[1]; j-- {
				people[j] = people[j-1]
			}
			people[p[1]] = temp
		}
	}
	return people
}

//func findMinArrowShots(points [][]int) int {
//	slices.SortFunc(points, func(a, b []int) int {
//对右边界排序
//		return a[1] - b[1]
//	})
//	prev := points[0]
//	ans := 0
//	for i := 1; i < len(points); i++ {
//		cur := points[i]
//		if cur[0] > prev[1] {
//			ans++
//			prev = cur
//		}
//	}
//	return ans
//}

//func findMinArrowShots(points [][]int) int {
//	slices.SortFunc(points, func(a, b []int) int {
//		//对左边界排序
//		return a[0] - b[0]
//	})
//	prev := points[0]
//	cnt := 0
//	for i := 1; i < len(points); i++ {
//		cur := points[i]
//		//与prev代表的气球没有重叠部分
//		if cur[0] > prev[1] {
//			cnt++
//			prev = cur
//		} else {
//			//cur被prev所包含 prev需要更新为范围更小的那一个
//			if cur[1] < prev[1] {
//				prev = cur
//			}
//		}
//	}
//	return cnt + 1
//}

func findMinArrowShots(points [][]int) int {
	slices.SortFunc(points, func(a, b []int) int {
		//对左边界排序
		return a[0] - b[0]
	})
	ans := 0
	right := points[0][1]
	for i := 1; i < len(points); i++ {
		cur := points[i]
		if cur[0] > right { //没有重叠部分
			ans++
			right = cur[1]
		} else { //有重叠部分
			right = min(right, cur[1])
		}
	}
	return ans + 1
}
func eraseOverlapIntervals(intervals [][]int) int {
	slices.SortFunc(intervals, func(a, b []int) int {
		return a[0] - b[0]
	})
	ans := 0
	right := intervals[0][1]
	for i := 1; i < len(intervals); i++ {
		cur := intervals[i]
		if cur[0] >= right { //区间没有重叠
			right = cur[1]
		} else {
			right = min(right, cur[1])
			ans++
		}
	}
	return ans
}

// 维护每一个字母的最后一个位置
func partitionLabels(s string) (ans []int) {
	lastAppearance := make([]int, 26)
	for i := 0; i < len(s); i++ {
		lastAppearance[s[i]-'a'] = i
	}
	for i := 0; i < len(s); {
		j := i
		right := lastAppearance[s[i]-'a']
		for ; j <= right; j++ {
			right = max(right, lastAppearance[s[j]-'a'])
		}
		ans = append(ans, j-i)
		i = j
	}
	return
}

// 合并区间
func merge2(intervals [][]int) (ans [][]int) {
	slices.SortFunc(intervals, func(a, b []int) int {
		return a[0] - b[0]
	})
	prev := intervals[0]
	for i := 1; i < len(intervals); i++ {
		cur := intervals[i]
		if cur[0] > prev[0] {
			ans = append(ans, prev)
		} else {
			prev[1] = max(prev[1], cur[1])
		}
	}
	return
}

// 单调递增的数字
// 贪心：从某一位开始后面都变成9
func monotoneIncreasingDigits(n int) (ans int) {
	ints := make([]int, 0)
	for ; n > 0; n /= 10 {
		ints = append(ints, n%10)
	}
	slices.Reverse(ints)
	flag := len(ints)
	for i := len(ints) - 1; i >= 1; i-- {
		if ints[i] < ints[i-1] {
			ints[i-1] -= 1
			flag = i
		}
	}
	for i := flag; i < len(ints); i++ {
		ints[i] = 9
	}
	for i := 0; i < len(ints); i++ {
		ans = 10*ans + ints[i]
	}
	return
}

func minCostClimbingStairs(cost []int) int {
	dp := make([]int, len(cost)+1)
	for i := 2; i <= len(cost); i++ {
		dp[i] = min(dp[i-1]+cost[i-1], dp[i-2]+cost[i-2])
	}
	return dp[len(cost)]
}

func uniquePaths(m int, n int) int {
	dp := make([][]int, m)
	for i, _ := range dp {
		dp[i] = make([]int, n)
	}
	for i := 0; i < m; i++ {
		dp[i][0] = 1
	}
	for i := 0; i < n; i++ {
		dp[0][i] = 1
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			dp[i][j] = dp[i-1][j] + dp[i][j-1]
		}
	}
	return dp[m-1][n-1]
}

func uniquePathsWithObstacles(obstacleGrid [][]int) int {
	m, n := len(obstacleGrid), len(obstacleGrid[0])
	dp := make([][]int, m)
	for i, _ := range dp {
		dp[i] = make([]int, n)
	}
	for i := 0; i < m && obstacleGrid[i][0] == 0; i++ {
		dp[i][0] = 1
	}
	for i := 0; i < n && obstacleGrid[0][i] == 0; i++ {
		dp[0][i] = 1
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			if obstacleGrid[i][j] == 0 {
				dp[i][j] = dp[i-1][j] + dp[i][j-1]
			}
		}
	}
	return dp[m-1][n-1]
}
func integerBreak(n int) int {
	dp := make([]int, n+1)
	dp[2] = 1
	for i := 3; i <= n; i++ {
		for j := 1; j < i; j++ {
			//dp[j] 拆分j时的最大乘积
			//j 不拆分时 就是j本身
			dp[i] = max(dp[i], max(dp[j], j)*(i-j))
		}
	}
	return dp[n]
}

// 一颗二叉搜索树的左子树和右子树都是二叉搜索树
func numTrees(n int) int {
	dp := make([]int, n+1)
	dp[0] = 1
	dp[1] = 1
	for i := 2; i <= n; i++ {
		for j := 0; j < i; j++ {
			//固定一个节点
			//假设左子树j个节点 右子树i-j-1个节点 总结点数为i个
			//左边j个节点组成的二叉搜索树有dp[j]种
			//右边同理
			dp[i] += dp[j] * dp[i-j-1]
		}
	}
	return dp[n]
}

//背包问题之01背包 一个物品只能使用一次
//dp[i][j]: 任取0-i的物品装满容量为j的背包的最大价值
//dp[i][j]=max(dp[i-1][j],dp[i-1][j-weight[i]]+value[i])
//dp[i-1][j]: 不取物品i装满容量为j的背包的最大价值
//dp[i-1][j-weight[i]]: 不取物品i时装满容量为j-weight[i]的背包的最大价值
//dp[i-1][j-weight[i]]+value[i]: 取物品i装满容量为j的背包的最大价值
//在使用二维dp数组时 需要初始化第一行和第一列
//且使用二维dp数组时 遍历背包和物品的顺序可以颠倒
//一维dp数组 dp[j]=max(dp[j],dp[j-weight[i]]+value[i])
//必须先遍历物品再遍历背包 且背包需要倒序遍历
//为什么需要倒序遍历: 确保物品只被添加一次
//递推公式中更大的j由之前的j推到得来
//若正序遍历会反复添加物品 例如容量为2的背包 正序遍历会两次添加重量为1的物品
// 对应背包容量为0时 初始化为0
// 对应物品0时根据物品0重量与当前背包容量的关系确定如何初始化

// 分割等和子集：能否装满容量为元素和1/2的背包
// 01背包问题的一种容量和价值等值的应用
func canPartition(nums []int) bool {
	var sum int
	for _, num := range nums {
		sum += num
	}
	if sum%2 != 0 {
		return false
	}
	//容量为target的背包所能装的最大价值是否是target
	target := sum / 2
	dp := make([]int, target+1)
	for i := 0; i < len(nums); i++ {
		for j := target; j >= nums[i]; j-- {
			dp[j] = max(dp[j], dp[j-nums[i]]+nums[i])
		}
	}
	return dp[target] == target
}

// 最后一块石头的重量
func lastStoneWeightII(stones []int) int {
	//[2,7,4,1,8,1] 总和为23
	//装满容量为12/11的背包的最大价值x
	//return |sum-2x|
	var sum int
	for _, num := range stones {
		sum += num
	}
	target := sum / 2
	dp := make([]int, target+1)
	for i := 0; i < len(stones); i++ {
		for j := target; j >= stones[i]; j-- {
			dp[j] = max(dp[j], dp[j-stones[i]]+stones[i])
		}
	}
	return Abs(sum - 2*dp[target])
}

// 回溯解法
//func findTargetSumWays(nums []int, target int) (ans int) {
//	var dfs func(start, sum int)
//	dfs = func(start, sum int) {
//		if start == len(nums) {
//			if sum == target {
//				ans++
//			}
//			return
//		}
//		//加号或者减号
//		dfs(start+1, sum+nums[start])
//		dfs(start+1, sum-nums[start])
//	}
//	dfs(0, 0)
//	return
//}

// 动态规划解法
// 将数组分为两个集合left和right
// left+right=sum left-right=target
// left=(sum+target)/2
func findTargetSumWays(nums []int, target int) int {
	var sum int
	for _, num := range nums {
		sum += num
	}
	if sum+target < 0 || (sum+target)%2 != 0 {
		return 0
	}
	left := (sum + target) / 2
	//dp[i]: 装满容量为i的背包有多少种方法
	dp := make([]int, left+1)
	dp[0] = 1
	for i := 0; i < len(nums); i++ {
		for j := left; j >= nums[i]; j-- {
			//装满j-nums[i]的背包有多少种方法即在确定装入nums[i]时装满j的背包有多少种方法
			//后续动态规划问题中xxx有多少种方法经常出现的递推公式
			dp[j] += dp[j-nums[i]]
		}
	}
	return dp[left]
}

// 相当于放入容量m和n的背包的最大物品数量
func findMaxForm(strs []string, m int, n int) int {
	var oneCount func(str string) int
	oneCount = func(str string) (cnt int) {
		for _, char := range str {
			if char == '1' {
				cnt++
			}
		}
		return
	}
	dp := make([][]int, m+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, n+1)
	}
	for i := 0; i < len(strs); i++ {
		oneCnt := oneCount(strs[i])
		zeroCnt := len(strs[i]) - oneCnt
		for j := m; j >= zeroCnt; j-- {
			for k := n; k >= oneCnt; k-- {
				dp[j][k] = max(dp[j][k], dp[j-zeroCnt][k-oneCnt]+1)
			}
		}
	}
	return dp[m][n]
}

// 背包问题之完全背包：每个物品可以使用无数次
// 对背包的遍历改为正序遍历即可 倒序遍历就是为了限制每一个物品只能使用一次
// 且对物品和背包的便利没有先后顺序要求
func change(amount int, coins []int) int {
	//dp[i] 表示组合金额i的组合数
	dp := make([]int, amount+1)
	dp[0] = 1
	for i := 0; i < len(coins); i++ {
		for j := coins[i]; j <= amount; j++ {
			dp[j] += dp[j-coins[i]]
		}
	}
	return dp[amount]
}
func combinationSum4(nums []int, target int) int {
	dp := make([]int, target+1)
	dp[0] = 1
	for j := 0; j <= target; j++ {
		for i := 0; i < len(nums); i++ {
			if j >= nums[i] {
				dp[j] += dp[j-nums[i]]
			}
		}
	}
	return dp[target]
}

// 组成amount所需的最小钱币数量
func coinChange(coins []int, amount int) int {
	if amount == 0 {
		return 0
	}
	dp := make([]int, amount+1)
	dp[0] = 0
	for i := 1; i < len(dp); i++ {
		dp[i] = math.MaxInt
	}
	for i := 0; i < len(coins); i++ {
		for j := coins[i]; j <= amount; j++ {
			//dp[j-coins[i]] == math.MaxInt 表示没有组成j-coins[i]的组合
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

// 完全平方数：完全背包
func numSquares(n int) int {
	//squares即物品
	squares := make([]int, 0, 100)
	for i := 1; i <= 100; i++ {
		squares = append(squares, i*i)
	}
	//dp[i]表示和为i的完全平方数的最少数量
	dp := make([]int, n+1)
	for i := 1; i < len(dp); i++ {
		dp[i] = i
	}
	for i := 0; i < len(squares); i++ {
		for j := squares[i]; j <= n; j++ {
			dp[j] = min(dp[j], dp[j-squares[i]]+1)
		}
	}
	return dp[n]
}

// 单词拆分
func wordBreak(s string, wordDict []string) bool {
	n := len(s)
	dp := make([]bool, n)
	//这里一开始遍历顺序写反了
	//原因是这里更类似于排列问题 因为组成单词的先后顺序是有意义的
	for j := 0; j < n; j++ {
		for i := 0; i < len(wordDict); i++ {
			if j+len(wordDict[i]) <= n {
				if (j == 0 || dp[j-1]) && s[j:j+len(wordDict[i])] == wordDict[i] {
					dp[j+len(wordDict[i])-1] = true
				}
			}
		}
	}
	return dp[n-1]
}

// 打家劫舍
func rob(nums []int) int {
	if len(nums) == 1 {
		return nums[0]
	}
	dp := make([]int, len(nums))
	dp[0] = nums[0]
	dp[1] = max(nums[0], nums[1])
	for i := 2; i < len(nums); i++ {
		//偷或者不偷
		dp[i] = max(dp[i-1], dp[i-2]+nums[i])
	}
	return dp[len(nums)-1]
}

// 打家劫舍2
func rob2(nums []int) int {
	if len(nums) == 1 {
		return nums[0]
	}
	//每一个dp[i]对应于两种状态：偷第一家和不偷第一家
	dp := make([][2]int, len(nums))
	dp[0][0] = nums[0]
	dp[0][1] = 0
	dp[1][0] = nums[0]
	dp[1][1] = nums[1]
	for i := 2; i < len(nums); i++ {
		dp[i][0] = dp[i-1][0]
		dp[i][1] = max(dp[i-1][1], dp[i-2][1]+nums[i])
		if i != len(nums)-1 {
			dp[i][0] = max(dp[i][0], dp[i-2][0]+nums[i])
		}
	}
	return max(dp[len(nums)-1][0], dp[len(nums)-1][1])
}

// 二叉树的直径
// 在每一个节点拐弯的直径：左子树最大深度+右子树最大深度
// 遍历的过程中不断求最大值
func diameterOfBinaryTree(root *TreeNode) (ans int) {
	var dfs func(node *TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		left := dfs(node.Left)
		right := dfs(node.Right)
		ans = max(ans, left+right)
		//当前节点的最大深度
		return max(left, right) + 1
	}
	dfs(root)
	return
}

// 最大路径和
// 核心在于递归返回的时候只能返回较大的一边（两边都是负数则不考虑）
// 而对每一个节点求最大值答案的时候可以两边都考虑
func maxPathSum(root *TreeNode) int {
	ans := math.MinInt
	var dfs func(node *TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		left := dfs(node.Left)
		right := dfs(node.Right)
		val := node.Val
		//返回所有可能情况下的最大值
		ans = max(ans, val, val+left, val+right, val+left+right)
		//只返回一边 如果两边都是负数 直接返回当前节点作为最大路径和
		return max(val, val+left, val+right)
	}
	dfs(root)
	return ans
}
