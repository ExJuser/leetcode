package sliding_window

import (
	"math"
	"slices"
	"strconv"
)

// 1456. 定长子串中元音的最大数目 定长滑动窗口
func maxVowels(s string, k int) int {
	vowels := []byte{'a', 'i', 'e', 'o', 'u'}
	var count int
	for i := 0; i < k; i++ {
		if slices.Contains(vowels, s[i]) {
			count++
		}
	}
	ans := count
	for i := 0; i < len(s)-k; i++ {
		if slices.Contains(vowels, s[i]) {
			count--
		}
		if slices.Contains(vowels, s[i+k]) {
			count++
		}
		ans = max(ans, count)
	}
	return ans
}

// 2269. 找到一个数字的 K 美丽值 定长滑动窗口
func divisorSubstrings(num int, k int) int {
	numStr := strconv.Itoa(num)
	mod := int(math.Pow10(k - 1))
	var val int
	for i := 0; i < k; i++ {
		val = val*10 + int(numStr[i]-'0')
	}
	var cnt int
	if num%val == 0 {
		cnt = 1
	}
	for i := 0; i < len(numStr)-k; i++ {
		val = (val%mod)*10 + int(numStr[i+k]-'0')
		if val != 0 && num%val == 0 {
			cnt++
		}
	}
	return cnt
}

// 1984. 学生分数的最小差值 定长滑动窗口
func minimumDifference(nums []int, k int) int {
	slices.Sort(nums)
	ans := math.MaxInt
	for i := 0; i < len(nums)-k+1; i++ {
		ans = min(ans, nums[i+k-1]-nums[i])
	}
	return ans
}

// 643. 子数组最大平均数 I
func findMaxAverage(nums []int, k int) float64 {
	var sum int
	for i := 0; i < k; i++ {
		sum += nums[i]
	}
	ans := float64(sum) / float64(k)
	for i := 0; i < len(nums)-k; i++ {
		sum -= nums[i] - nums[i+k]
		ans = max(ans, float64(sum)/float64(k))
	}
	return ans
}

// 1493. 删掉一个元素以后全为 1 的最长子数组
func longestSubarray(nums []int) int {
	var left, ans, zeroCnt int
	for right := 0; right < len(nums); right++ {
		if nums[right] == 0 {
			zeroCnt++
		}
		for ; zeroCnt > 1; left++ {
			if nums[left] == 0 {
				zeroCnt--
			}
		}
		ans = max(ans, right-left+1-zeroCnt)
		if zeroCnt == 0 {
			ans -= 1
		}
	}
	return ans
}

// 2730. 找到最长的半重复子字符串
func longestSemiRepetitiveSubstring(s string) int {
	var left, count int
	ans := 1
	for right := 1; right < len(s); right++ {
		if s[right] == s[right-1] {
			count++
		}
		for ; count > 1; left++ {
			if s[left] == s[left+1] {
				count--
			}
		}
		ans = max(ans, right-left+1)
	}
	return ans
}
