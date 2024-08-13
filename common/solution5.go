package common

import (
	"slices"
	"strconv"
	"strings"
)

// 2684. 矩阵中移动的最大次数 普通二维dp
func maxMoves(grid [][]int) int {
	dp := make([][]int, len(grid))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(grid[i]))
	}
	var ans int
	for j := 0; j < len(dp[0]); j++ {
		for i := 0; i < len(grid); i++ {
			if dp[i][j] == j { //可以到达当前格子
				//尝试三个方向
				if j+1 < len(grid[0]) {
					if dp[i][j+1] != j+1 {
						if grid[i][j+1] > grid[i][j] {
							dp[i][j+1] = j + 1
							ans = max(ans, dp[i][j+1])
						}
					}
					if i-1 >= 0 && dp[i-1][j+1] != j+1 {
						if grid[i-1][j+1] > grid[i][j] {
							dp[i-1][j+1] = j + 1
							ans = max(ans, dp[i-1][j+1])
						}
					}
					if i+1 < len(grid) && dp[i+1][j+1] != j+1 {
						if grid[i+1][j+1] > grid[i][j] {
							dp[i+1][j+1] = j + 1
							ans = max(ans, dp[i+1][j+1])
						}
					}
				}
			}
		}
	}
	return ans
}

// 464. 我能赢吗 博弈 状压dp
func canIWin(maxChoosableInteger int, desiredTotal int) bool {
	if desiredTotal == 0 {
		return true
	}
	if (maxChoosableInteger*(1+maxChoosableInteger))/2 < desiredTotal {
		return false
	}
	//dfs 当前可以选择的数字 还差多少满足累计和 返回能否稳赢
	//可以选择的数字使用状态位表示 挂一个缓存表
	state := (1 << (maxChoosableInteger + 1)) - 2
	cache := make(map[int]bool)
	var dfs func(state int, desired int) bool
	dfs = func(state int, desired int) bool {
		if desired <= 0 {
			return false
		}
		if v, ok := cache[state]; ok {
			return v
		}
		for i := 1; i <= maxChoosableInteger; i++ {
			//这个数字不能被使用过
			if state&(1<<i) != 0 {
				//对手不能赢
				if !dfs(state^(1<<i), desired-i) {
					cache[state] = true
					return true
				} else {
					cache[state^(1<<i)] = true
				}
			}
		}
		cache[state] = false
		return false
	}
	return dfs(state, desiredTotal)
}

// 698. 划分为k个相等的子集 状压dp+缓存表
func canPartitionKSubsets(nums []int, k int) bool {
	n := len(nums)
	var sum int
	for _, num := range nums {
		sum += num
	}
	if sum%k != 0 {
		return false
	}
	limit := sum / k
	//所有元素都可以使用的状态
	state := (1 << n) - 1
	//缓存每个状态的结果
	cache := make([]int, 1<<n)
	var dfs func(state, cur, rest int, cache []int) bool
	dfs = func(state, cur, rest int, cache []int) bool {
		if rest == 0 {
			return true
		}
		if cache[state] != 0 { //之前计算过
			return cache[state] == 1
		}
		var ans bool
		for i := 0; i < n; i++ {
			//还没被用过而且不超出范围
			if state&(1<<i) != 0 && cur+nums[i] <= limit {
				if cur+nums[i] == limit {
					ans = dfs(state^(1<<i), 0, rest-1, cache)
				} else {
					ans = dfs(state^(1<<i), cur+nums[i], rest, cache)
				}
				if ans {
					break
				}
			}
		}
		if ans {
			cache[state] = 1
		} else {
			cache[state] = -1
		}
		return ans
	}
	return dfs(state, 0, k, cache)
}

// 698.火柴拼正方形 状压dp+缓存表
// 缓存表的用法 在dfs中作为参数携带
// 先判断base case 再判断是否有缓存 返回前设置缓存的值
func makesquare(matchsticks []int) bool {
	n := len(matchsticks)
	var sum int
	for _, num := range matchsticks {
		sum += num
	}
	if sum%4 != 0 {
		return false
	}
	limit := sum / 4
	//所有元素都可以使用的状态
	state := (1 << n) - 1
	//缓存每个状态的结果
	cache := make([]int, 1<<n)
	var dfs func(state, cur, rest int, cache []int) bool
	dfs = func(state, cur, rest int, cache []int) bool {
		if rest == 0 {
			return true
		}
		if cache[state] != 0 { //之前计算过
			return cache[state] == 1
		}
		var ans bool
		for i := 0; i < n; i++ {
			//还没被用过而且不超出范围
			if state&(1<<i) != 0 && cur+matchsticks[i] <= limit {
				if cur+matchsticks[i] == limit {
					ans = dfs(state^(1<<i), 0, rest-1, cache)
				} else {
					ans = dfs(state^(1<<i), cur+matchsticks[i], rest, cache)
				}
				if ans {
					break
				}
			}
		}
		if ans {
			cache[state] = 1
		} else {
			cache[state] = -1
		}
		return ans
	}
	return dfs(state, 0, 4, cache)
}

// 最长相同子序列 初始化比较麻烦
func maxUncrossedLines(nums1 []int, nums2 []int) int {
	dp := make([][]int, len(nums1))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(nums2))
	}
	for i := 0; i < len(dp); i++ {
		if nums1[i] == nums2[0] {
			for ; i < len(dp); i++ {
				dp[i][0] = 1
			}
			break
		}
	}
	for i := 0; i < len(dp[0]); i++ {
		if nums2[i] == nums1[0] {
			for ; i < len(dp[0]); i++ {
				dp[0][i] = 1
			}
			break
		}
	}
	for i := 1; i < len(nums1); i++ {
		for j := 1; j < len(nums2); j++ {
			if nums1[i] == nums2[j] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
			}
		}
	}
	return dp[len(nums1)-1][len(nums2)-1]
}

// 179. 最大数
func largestNumber(nums []int) string {
	slices.SortFunc(nums, func(a, b int) int {
		strA := strconv.Itoa(a)
		strB := strconv.Itoa(b)
		return strings.Compare(strB+strA, strA+strB)
	})
	sb := strings.Builder{}
	for _, num := range nums {
		sb.WriteString(strconv.Itoa(num))
	}
	str := sb.String()
	//去掉前导0 找到第一个非零索引
	var i int
	if str[0] == '0' && len(str) > 1 {
		for i < len(str) && str[i] == '0' {
			i++
		}
	}
	if i == len(str) {
		return "0"
	}
	return str[i:]
}
