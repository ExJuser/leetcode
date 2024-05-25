package main

func maxSatisfied(customers []int, grumpy []int, minutes int) int {
	var ans, cnt int
	for i := 0; i < len(customers); i++ {
		if grumpy[i] == 0 {
			cnt += customers[i]
		}
	}
	for i := 0; i < minutes && i < len(grumpy); i++ {
		if grumpy[i] == 1 {
			cnt += customers[i]
		}
	}
	ans = cnt
	for i := 1; i < len(customers)-minutes+1; i++ {
		if grumpy[i-1] == 1 {
			cnt -= customers[i-1]
		}
		if grumpy[i+minutes-1] == 1 {
			cnt += customers[i+minutes-1]
		}
		ans = max(ans, cnt)
	}
	return ans
}

func maxSum(nums []int, m int, k int) int64 {
	mp := make(map[int]int)
	var sum, ans int64
	for i := 0; i < k; i++ {
		mp[nums[i]]++
		sum += int64(nums[i])
	}
	if len(mp) >= m {
		ans = sum
	}
	for i := 1; i < len(nums)-k+1; i++ {
		mp[nums[i-1]]--
		mp[nums[i+k-1]]++
		sum -= int64(nums[i-1] - nums[i+k-1])
		if cnt, ok := mp[nums[i-1]]; ok && cnt == 0 {
			delete(mp, nums[i-1])
		}
		if len(mp) >= m {
			ans = max(ans, sum)
		}
	}
	return ans
}
