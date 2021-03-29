package base62

import (
	"math/rand"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestBase62(t *testing.T) {
	wg := sync.WaitGroup{}
	for i := 0; i < 1024; i++ {
		d := []byte(random(i))
		wg.Add(1)
		go func(i int) {
			assert.Equal(t, d, StdEncoding.DecodeString(StdEncoding.EncodeToString(d)))
			assert.Equal(t, d, FlipEncoding.DecodeString(FlipEncoding.EncodeToString(d)))
			assert.Equal(t, d, ShiftEncoding.DecodeString(ShiftEncoding.EncodeToString(d)))
			assert.Equal(t, d, FlipShiftEncoding.DecodeString(FlipShiftEncoding.EncodeToString(d)))
			wg.Done()
		}(i)
	}

	wg.Wait()
}

func random(l int) string {
	rand.Seed(time.Now().UnixNano())
	chars := []rune("ABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖ" +
		"abcdefghijklmnopqrstuvwxyzåäö" +
		"0123456789")
	var b strings.Builder
	for i := 0; i < l; i++ {
		b.WriteRune(chars[rand.Intn(len(chars))])
	}
	return b.String()
}
