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
