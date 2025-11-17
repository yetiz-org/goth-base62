// Package base62 provides efficient Base62 encoding with multiple encoding variants.
//
// Base62 encoding uses 62 characters (0-9, A-Z, a-z) for URL-safe, readable output.
// Six encoding variants are supported:
//   - StdEncoding: Standard encoding with length preservation (default)
//   - FlipEncoding: Reversed character order with length preservation (default)
//   - ShiftEncoding: Random shift for additional entropy
//   - FlipShiftEncoding: Combined reverse and shift
//   - StdLengthEncoding: Alias for StdEncoding (deprecated, use StdEncoding)
//   - FlipLengthEncoding: Alias for FlipEncoding (deprecated, use FlipEncoding)
//
// Important behaviors:
//   - StdEncoding and FlipEncoding preserve original length including leading zeros
//   - ShiftEncoding and FlipShiftEncoding preserve original length with random encoding
//   - All encoders are goroutine-safe
//
// For AES encryption/decryption or any use case requiring exact byte length preservation,
// use StdEncoding or FlipEncoding (both preserve length by default).
package base62

import (
	"errors"
	"math/big"
	"math/rand"
	"strings"
	"sync"
)

const (
	encodeStd  = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
	Base62Size = 62
)

var (
	baseInt = Base62Size
	base    = big.NewInt(int64(baseInt))

	// StdEncoding is the standard Base62 encoder with length preservation
	StdEncoding = NewEncoding(encodeStd).WithLength(true)

	// FlipEncoding reverses character order with length preservation
	FlipEncoding = NewEncoding(encodeStd).Direction(true).WithLength(true)

	// ShiftEncoding adds random shift for entropy
	ShiftEncoding = NewEncoding(encodeStd).Shift(true)

	// FlipShiftEncoding combines reverse and shift
	FlipShiftEncoding = NewEncoding(encodeStd).Shift(true).Direction(true)

	StdLengthEncoding  = NewEncoding(encodeStd).WithLength(true)
	FlipLengthEncoding = NewEncoding(encodeStd).Direction(true).WithLength(true)

	bigIntPool = sync.Pool{
		New: func() interface{} {
			return &big.Int{}
		},
	}
)

type Encoding struct {
	encode     [Base62Size]byte
	decodeMap  [256]byte
	forward    bool
	shift      bool
	withLength bool
}

// ErrInvalidCharacter indicates an invalid character during decoding
var ErrInvalidCharacter = errors.New("base62: invalid character")

// NewEncoding creates a new Base62 encoder with the given alphabet.
func NewEncoding(encoder string) *Encoding {
	if len(encoder) != Base62Size {
		panic("encoding alphabet is not 62-bytes long")
	}

	seen := make(map[byte]bool, Base62Size)
	for i := 0; i < len(encoder); i++ {
		if encoder[i] == '\n' || encoder[i] == '\r' {
			panic("encoding alphabet contains newline character")
		}
		if seen[encoder[i]] {
			panic("encoding alphabet contains duplicate character")
		}
		seen[encoder[i]] = true
	}

	e := new(Encoding)
	copy(e.encode[:], encoder)

	for i := 0; i < len(e.decodeMap); i++ {
		e.decodeMap[i] = 0xFF
	}
	for i := 0; i < len(encoder); i++ {
		e.decodeMap[encoder[i]] = byte(i)
	}

	return e
}

// Direction sets encoding direction.
func (enc *Encoding) Direction(forward bool) *Encoding {
	enc.forward = forward
	return enc
}

// Shift enables shift encoding.
func (enc *Encoding) Shift(shift bool) *Encoding {
	enc.shift = shift
	return enc
}

// WithLength enables length preservation.
func (enc *Encoding) WithLength(withLength bool) *Encoding {
	enc.withLength = withLength
	return enc
}

// EncodeToString encodes bytes to Base62 string
func (enc *Encoding) EncodeToString(src []byte) string {
	if src == nil || len(src) == 0 {
		return ""
	}

	sl := len(src)
	num := bigIntPool.Get().(*big.Int)
	defer bigIntPool.Put(num)
	num.SetInt64(0)

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
	} else if enc.withLength {
		prefixed := make([]byte, sl+1)
		prefixed[0] = 0xFF
		copy(prefixed[1:], src)
		num.SetBytes(prefixed)
	} else {
		num.SetBytes(src)
	}

	return enc.doEncode(num, shift).String()
}

func (enc *Encoding) doEncode(num *big.Int, shift int) *strings.Builder {
	estimatedLen := (len(num.Bytes()) * 11 / 8) + 2
	if shift >= 0 {
		estimatedLen++
	}
	builder := new(strings.Builder)
	builder.Grow(estimatedLen)

	if shift >= 0 {
		builder.WriteByte(enc.encode[shift])
	}

	mod := bigIntPool.Get().(*big.Int)
	defer bigIntPool.Put(mod)

	for num.Sign() != 0 {
		mod.Mod(num, base)
		builder.WriteByte(enc.encode[mod.Int64()])
		num.Div(num, base)
	}

	if enc.forward {
		rtn := new(strings.Builder)
		bs := builder.String()
		l := len(bs)
		rtn.Grow(l)
		for i := l - 1; i >= 0; i-- {
			rtn.WriteByte(bs[i])
		}
		return rtn
	}

	return builder
}

// DecodeString decodes Base62 string to bytes.
func (enc *Encoding) DecodeString(s string) []byte {
	result, err := enc.decodeStringInternal(s, false)
	if err != nil {
		panic(err)
	}
	return result
}

// DecodeStringStrict decodes Base62 string to bytes with strict validation.
func (enc *Encoding) DecodeStringStrict(s string) ([]byte, error) {
	return enc.decodeStringInternal(s, true)
}

func (enc *Encoding) decodeStringInternal(s string, strict bool) ([]byte, error) {
	if s == "" {
		return []byte{}, nil
	}

	for i := 0; i < len(s); i++ {
		if s[i] >= 128 {
			if strict {
				return nil, ErrInvalidCharacter
			} else {
				panic("runtime error: index out of range")
			}
		}
		if strict && enc.decodeMap[s[i]] == 0xFF {
			return nil, ErrInvalidCharacter
		}
	}

	if enc.forward {
		bs := make([]byte, len(s))
		for i := 0; i < len(s); i++ {
			bs[i] = s[len(s)-1-i]
		}
		s = string(bs)
	}

	shift := 0
	if enc.shift {
		if len(s) == 0 {
			return []byte{}, nil
		}
		shift = int(enc.decodeMap[s[0]])
		s = s[1:]
	}

	num := bigIntPool.Get().(*big.Int)
	defer bigIntPool.Put(num)
	num.SetInt64(0)

	tmp := bigIntPool.Get().(*big.Int)
	defer bigIntPool.Put(tmp)

	exp := bigIntPool.Get().(*big.Int)
	defer bigIntPool.Put(exp)

	for i, e := range s {
		tmp.SetInt64(int64(enc.decodeMap[e]))
		exp.Exp(base, big.NewInt(int64(i)), nil)
		tmp.Mul(tmp, exp)
		num.Add(num, tmp)
	}

	bs := num.Bytes()
	sl := len(bs)

	if enc.shift {
		if sl <= 1 {
			return []byte{}, nil
		}
		shifted := make([]byte, sl-1)
		sb := byte(shift)
		for i := 1; i < sl; i++ {
			shifted[(i+shift)%(sl-1)] = bs[i] ^ sb
		}
		return shifted, nil
	}

	if enc.withLength {
		if sl > 0 && bs[0] == 0xFF {
			return bs[1:], nil
		}
		return bs, nil
	}

	return bs, nil
}
