package main

import (
	"fmt"
	"net/http"
)

func main() {
	http.HandleFunc("/ping/", func(writer http.ResponseWriter, request *http.Request) {
		_, _ = writer.Write([]byte("pong"))
		fmt.Println(request.URL)
		fmt.Println(request.Method)
		fmt.Println(request.Host)
		fmt.Println(request.Body)
	})
	_ = http.ListenAndServe(":8091", nil)
}
