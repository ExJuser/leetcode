package main

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
