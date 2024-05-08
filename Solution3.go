package main

import (
	"container/heap"
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
