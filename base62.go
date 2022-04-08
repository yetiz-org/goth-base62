package base62

import (
	"math/big"
	"math/rand"
	"strings"
	"time"
)

const (
	encodeStd = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)

var baseInt = 62
var base = big.NewInt(int64(baseInt))
var StdEncoding = NewEncoding(encodeStd)
var FlipEncoding = NewEncoding(encodeStd).Direction(true)
var ShiftEncoding = NewEncoding(encodeStd).Shift(true)
var FlipShiftEncoding = NewEncoding(encodeStd).Shift(true).Direction(true)

type Encoding struct {
	encode    [62]byte
	decodeMap [256]byte
	forward   bool
	shift     bool
}

func NewEncoding(encoder string) *Encoding {
	if len(encoder) != 62 {
		panic("encoding alphabet is not 62-bytes long")
	}
	for i := 0; i < len(encoder); i++ {
		if encoder[i] == '\n' || encoder[i] == '\r' {
			panic("encoding alphabet contains newline character")
		}
	}

	e := new(Encoding)
	copy(e.encode[:], encoder)

	for i := 0; i < len(e.decodeMap); i++ {
		e.decodeMap[i] = 0xFF
	}
	for i := 0; i < len(encoder); i++ {
		e.decodeMap[encoder[i]] = byte(i)
	}

	rand.Seed(time.Now().UTC().UnixNano())
	return e
}

func (enc *Encoding) Direction(forward bool) *Encoding {
	enc.forward = forward
	return enc
}

func (enc *Encoding) Shift(shift bool) *Encoding {
	enc.shift = shift
	return enc
}

func (enc *Encoding) EncodeToString(src []byte) string {
	if src == nil || len(src) == 0 {
		return ""
	}

	sl := len(src)
	num := &big.Int{}
	shift := func() int {
		if !enc.shift {
			return -1
		}

		if sl == 1 {
			return 0
		}

		r := int(rand.Int31n(int32(sl)))
		return r % baseInt
	}()

	if enc.shift {
		shifted := make([]byte, sl+1)
		shifted[0] = 0xFF
		sb := byte(shift)
		for i := 1; i < sl+1; i++ {
			shifted[i] = src[(shift+i)%sl] ^ sb
		}

		num.SetBytes(shifted)
	} else {
		num.SetBytes(src)
	}

	return enc.doEncode(num, shift).String()
}

func (enc *Encoding) doEncode(num *big.Int, shift int) *strings.Builder {
	builder := new(strings.Builder)
	if shift >= 0 {
		builder.WriteByte(enc.encode[shift])
	}

	for num.Int64() != 0 {
		mod := enc.encode[new(big.Int).Mod(num, base).Int64()]
		num = num.Div(num, base)
		builder.WriteByte(mod)
	}

	if enc.forward {
		rtn := new(strings.Builder)
		bs := builder.String()
		l := len(bs)
		rd := make([]byte, l)
		for i := 0; i < l; i++ {
			rd[l-i-1] = bs[i]
		}

		rtn.Write(rd)
		return rtn
	}

	return builder
}

func (enc *Encoding) DecodeString(s string) []byte {
	if s == "" {
		return []byte{}
	}

	if enc.forward {
		bs := []byte(s)
		l := len(s)
		for i := 0; i < l/2; i++ {
			bs[i], bs[l-i-1] = bs[l-i-1], bs[i]
		}

		s = string(bs)
	}

	shift := 0
	if enc.shift {
		shift = int(enc.decodeMap[s[0]])
		s = s[1:]
	}

	num := &big.Int{}
	for i, e := range s {
		tmp := &big.Int{}
		num.Add(num, tmp.Mul(big.NewInt(int64(enc.decodeMap[e])), big.NewInt(0).Exp(base, big.NewInt(int64(i)), nil)))
	}

	bs := num.Bytes()
	sl := len(bs)

	if enc.shift {
		shifted := make([]byte, sl-1)
		sb := byte(shift)
		for i := 1; i < sl; i++ {
			shifted[(i+shift)%(sl-1)] = bs[i] ^ sb
		}

		return shifted
	}

	return bs
}
