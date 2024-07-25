package dp

import (
	"slices"
)

// 70. 爬楼梯
func climbStairs(n int) int {
	if n <= 2 {
		return n
	}
	dp1, dp2 := 1, 2
	for i := 3; i <= n; i++ {
		dp2, dp1 = dp1+dp2, dp2
	}
	return dp2
}

// 746. 使用最小花费爬楼梯
func minCostClimbingStairs(cost []int) int {
	dp1, dp2 := 0, 0
	for i := 2; i <= len(cost); i++ {
		dp2, dp1 = min(dp1+cost[i-2], dp2+cost[i-1]), dp2
	}
	return dp2
}

// 377. 组合总和 Ⅳ 没看出来是背包问题的做法
//
//	func combinationSum4(nums []int, target int) int {
//		//dpi 表示能组合成i的个数
//		//dpi = dp[i-1]+dp[i-2] :如果存在1和2
//		mp := make(map[int]struct{})
//		dp := make([]int, target+1)
//		for _, num := range nums {
//			mp[num] = struct{}{}
//			if num <= target {
//				dp[num] = 1
//			}
//		}
//
//		for i := 1; i < len(dp); i++ {
//			for k, _ := range mp {
//				if i-k >= 0 {
//					dp[i] += dp[i-k]
//				}
//			}
//		}
//		return dp[target]
//	}
//
// 377. 组合总和 Ⅳ 实际上是完全背包问题
func combinationSum4(nums []int, target int) int {
	//装满容量为i的背包有多少种方式
	//由于求的是排列数 因此对物品的遍历就要放在里面
	dp := make([]int, target+1)
	dp[0] = 1
	for i := 0; i <= target; i++ {
		for j := 0; j < len(nums); j++ {
			if i >= nums[j] {
				dp[i] += dp[i-nums[j]]
			}
		}
	}
	return dp[target]
}

// 2466. 统计构造好字符串的方案数
func countGoodStrings(low int, high int, zero int, one int) int {
	var mod int = 1e9 + 7
	dp := make([]int, high+1)
	dp[zero] += 1
	dp[one] += 1
	for i := min(one, zero); i <= high; i++ {
		if i >= zero {
			dp[i] = (dp[i] + dp[i-zero]) % mod
		}
		if i >= one {
			dp[i] = (dp[i] + dp[i-one]) % mod
		}
	}
	var ans int
	for i := low; i <= high; i++ {
		ans = (ans + dp[i]) % mod
	}
	return ans
}

// 740. 删除并获得点数
func deleteAndEarn(nums []int) int {
	rob := make([]int, slices.Max(nums)+1)
	for _, num := range nums {
		rob[num] += num
	}
	dp := make([]int, len(rob))
	if len(rob) == 1 {
		return rob[0]
	}
	dp[0] = rob[0]
	dp[1] = max(rob[0], rob[1])
	for i := 2; i < len(rob); i++ {
		dp[i] = max(dp[i-1], dp[i-2]+rob[i])
	}
	return dp[len(rob)-1]
}

func countHousePlacements(n int) int {
	var mod int = 1e9 + 7
	//放或者不放
	dp := make([][2]int, n)
	dp[0][0] = 1
	dp[0][1] = 1
	for i := 1; i < n; i++ {
		dp[i][0] = dp[i-1][1]
		dp[i][1] = (dp[i-1][0] + dp[i-1][1]) % mod
	}
	temp := (dp[n-1][0] + dp[n-1][1]) % mod
	return (temp * temp) % mod
}

func rob(nums []int) int {
	if len(nums) <= 2 {
		return slices.Max(nums)
	}
	//如果偷第一家：那么最后一家不能偷 相当于2~n-1的常规打家劫舍+nums[0]
	//如果不偷第一家：那么相当于1~n的常规打家劫舍
	return max(rob_(nums[1:]), rob_(nums[2:len(nums)-1])+nums[0])
}

func rob_(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	if len(nums) <= 2 {
		return slices.Max(nums)
	}
	dp1, dp2 := nums[0], max(nums[0], nums[1])
	for i := 2; i < len(nums); i++ {
		dp1, dp2 = dp2, max(dp1+nums[i], dp2)
	}
	return dp2
}

// 53. 最大子数组和
func maxSubArray(nums []int) int {
	ans := nums[0]
	for i := 1; i < len(nums); i++ {
		nums[i] = max(nums[i-1]+nums[i], nums[i])
		ans = max(ans, nums[i])
	}
	return ans
}

// 2606. 找到最大开销的子字符串
func maximumCostSubstring(s string, chars string, vals []int) int {
	//对于一个字符ch 如果它在chars中 它的价值就是chars中的对应
	valsMap := make(map[byte]int)
	for i, ch := range chars {
		valsMap[byte(ch)] = vals[i]
	}
	dp := make([]int, len(s))
	val, ok := valsMap[s[0]]
	if !ok {
		val = int(s[0] - 'a' + 1)
	}
	dp[0] = max(val, 0)
	ans := dp[0]
	for i := 1; i < len(dp); i++ {
		val, ok := valsMap[s[i]]
		if !ok {
			val = int(s[i] - 'a' + 1)
		}
		dp[i] = max(dp[i-1]+val, val, 0)
		ans = max(ans, dp[i])
	}
	return ans
}

type Number interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 | ~float32 | ~float64
}

func Abs[T Number](num T) T {
	if num < 0 {
		return -num
	}
	return num
}

// 1749. 任意子数组和的绝对值的最大值 二维dp
func maxAbsoluteSum(nums []int) int {
	//分别维护最大值和最小值 答案是绝对值更大的那个
	dp0 := nums[0]
	dp1 := nums[0]
	ans := Abs(nums[0])
	for i := 1; i < len(nums); i++ {
		dp0 = min(dp0+nums[i], nums[i])
		dp1 = max(dp1+nums[i], nums[i])
		ans = max(ans, Abs(dp0), Abs(dp1))
	}
	return ans
}

// LCR 166. 珠宝的最高价值
func jewelleryValue(frame [][]int) int {
	dp := make([][]int, len(frame))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(frame[i]))
	}
	dp[0][0] = frame[0][0]
	for i := 1; i < len(frame); i++ {
		dp[i][0] = dp[i-1][0] + frame[i][0]
	}
	for i := 1; i < len(frame[0]); i++ {
		dp[0][i] = dp[0][i-1] + frame[0][i]
	}
	for i := 1; i < len(frame); i++ {
		for j := 1; j < len(frame[i]); j++ {
			dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + frame[i][j]
		}
	}
	return dp[len(dp)-1][len(dp[0])-1]
}

// 62. 不同路径
func uniquePaths(m int, n int) int {

}
