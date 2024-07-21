package main

import (
	"fmt"
	"strconv"
	"strings"
)

//for debug

func totalFruit(fruits []int) int {
	mp := make(map[int]int)
	var left, ans int
	for right := 0; right < len(fruits); right++ {
		mp[fruits[right]]++
		for ; len(mp) > 2; left++ {
			mp[fruits[left]]--
			if mp[fruits[left]] == 0 {
				delete(mp, fruits[left])
			}
		}
		ans = max(ans, right-left+1)
	}
	return ans
}

func minWindow(s string, t string) string {
	var left int
	var check func(mp map[byte]int) bool
	check = func(mp map[byte]int) bool {
		for _, v := range mp {
			if v > 0 {
				return false
			}
		}
		return true
	}
	mp := make(map[byte]int)
	length := len(s) + 1
	ans := ""
	for _, ch := range t {
		mp[byte(ch)]++
	}
	for right := 0; right < len(s); right++ {
		mp[s[right]]--
		for ; check(mp); left++ {
			if right-left+1 < length {
				length = right - left + 1
				ans = s[left : right+1]
			}
			mp[s[left]]++
		}
	}
	return ans
}

func spiralOrder(matrix [][]int) (ans []int) {
	height, width := len(matrix), len(matrix[0])
	top, bottom, left, right := 0, height-1, 0, width-1
	for left < right && top < bottom {
		for i := left; i <= right; i++ {
			ans = append(ans, matrix[top][i])
		}
		top++
		for i := top; i <= bottom; i++ {
			ans = append(ans, matrix[i][right])
		}
		right--
		for i := right; i >= left; i-- {
			ans = append(ans, matrix[bottom][i])
		}
		bottom--
		for i := bottom; i >= top; i-- {
			ans = append(ans, matrix[i][left])
		}
		left++
	}
	//只剩一行
	if top == bottom {
		for i := left; i <= right; i++ {
			ans = append(ans, matrix[top][i])
		}
	} else if left == right {
		for i := top; i <= bottom; i++ {
			ans = append(ans, matrix[i][left])
		}
	}
	return
}

// 5. 最长回文子串 动态规划
func longestPalindrome(s string) string {
	dp := make([][]int, len(s))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(s))
	}
	//dp[i][j]可以由dp[i+1][j-1]推出
	//初始对角线：单个字符的字符串都是回文串
	for i := 0; i < len(s); i++ {
		dp[i][i] = 1
	}
	length := 1
	ans := s[0:1]
	//注意遍历的顺序问题
	for j := 1; j < len(s); j++ {
		for i := 0; i < j; i++ {
			if s[i] == s[j] && dp[i+1][j-1] == j-i-1 {
				dp[i][j] = dp[i+1][j-1] + 2
			} else {
				dp[i][j] = 0
			}
			if dp[i][j] > length {
				length = dp[i][j]
				ans = s[i : j+1]
			}
		}
	}
	return ans
}

// 93. 复原 IP 地址 回溯插入三个点/找到三个切割位置
func restoreIpAddresses(s string) (ans []string) {
	//如何判断是否合法：切割出来的数字处于0-255 不能有前导0
	var dfs func(pre, cur int, path []string)
	var check func(str string) bool
	check = func(str string) bool {
		//如果首位是0 那他必须是零本身
		if str[0] == '0' {
			return len(str) == 1
		}
		val, _ := strconv.Atoi(str)
		return val >= 0 && val <= 255
	}
	dfs = func(pre, cur int, path []string) {
		if cur == len(s) {
			if len(path) == 4 && cur == pre {
				ans = append(ans, strings.Join(path, "."))
			}
			return
		}
		//切割
		//判断是否合法
		if check(s[pre : cur+1]) {
			path = append(path, s[pre:cur+1])
			dfs(cur+1, cur+1, path)
			path = path[:len(path)-1]
		}
		//不切割
		dfs(pre, cur+1, path)
	}
	dfs(0, 0, []string{})
	return
}

func addStrings(num1 string, num2 string) string {
	carry := 0
	bytes1 := []byte(num1)
	bytes2 := []byte(num2)
	res := make([]byte, max(len(num1), len(num2)))
	index := len(res) - 1
	i, j := len(bytes1)-1, len(bytes2)-1
	for ; i >= 0 && j >= 0; i, j = i-1, j-1 {
		val := carry + int(bytes1[i]-'0') + int(bytes2[j]-'0')
		carry = val / 10
		val %= 10
		res[index] = byte(val) + '0'
		index--
	}
	for ; i >= 0; i-- {
		val := carry + int(bytes1[i]-'0')
		carry = val / 10
		val %= 10
		res[index] = byte(val) + '0'
		index--
	}
	for ; j >= 0; j-- {
		val := carry + int(bytes2[j]-'0')
		carry = val / 10
		val %= 10
		res[index] = byte(val) + '0'
		index--
	}
	if carry != 0 {
		return fmt.Sprintf("%d%s", carry, string(res))
	}
	return string(res)
}

func main() {
	fields := strings.Fields("  hello world!  ")
	fmt.Println(fields, len(fields))
}
