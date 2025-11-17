package base62

import (
	"bytes"
	"crypto/rand"
	"testing"
)

func TestStdEncodingLengthPreservation(t *testing.T) {
	tests := []struct {
		name  string
		input []byte
	}{
		{"empty", []byte{}},
		{"single zero", []byte{0x00}},
		{"leading zeros", []byte{0x00, 0x00, 0x01}},
		{"multiple leading zeros", []byte{0x00, 0x00, 0x00, 0x42}},
		{"all zeros", []byte{0x00, 0x00, 0x00}},
		{"no leading zeros", []byte{0x42, 0x43, 0x44}},
		{"mixed", []byte{0x00, 0x42, 0x00, 0x43}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encoded := StdEncoding.EncodeToString(tt.input)
			decoded := StdEncoding.DecodeString(encoded)

			if !bytes.Equal(decoded, tt.input) {
				t.Errorf("StdEncoding: input=%v, encoded=%s, decoded=%v", tt.input, encoded, decoded)
			}

			if len(tt.input) > 0 && len(decoded) != len(tt.input) {
				t.Errorf("StdEncoding: length not preserved: input_len=%d, decoded_len=%d", len(tt.input), len(decoded))
			}
		})
	}
}

func TestFlipEncodingLengthPreservation(t *testing.T) {
	tests := []struct {
		name  string
		input []byte
	}{
		{"empty", []byte{}},
		{"single zero", []byte{0x00}},
		{"leading zeros", []byte{0x00, 0x00, 0x01}},
		{"multiple leading zeros", []byte{0x00, 0x00, 0x00, 0x42}},
		{"all zeros", []byte{0x00, 0x00, 0x00}},
		{"no leading zeros", []byte{0x42, 0x43, 0x44}},
		{"mixed", []byte{0x00, 0x42, 0x00, 0x43}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encoded := FlipEncoding.EncodeToString(tt.input)
			decoded := FlipEncoding.DecodeString(encoded)

			if !bytes.Equal(decoded, tt.input) {
				t.Errorf("FlipEncoding: input=%v, encoded=%s, decoded=%v", tt.input, encoded, decoded)
			}

			if len(tt.input) > 0 && len(decoded) != len(tt.input) {
				t.Errorf("FlipEncoding: length not preserved: input_len=%d, decoded_len=%d", len(tt.input), len(decoded))
			}
		})
	}
}

func TestStdEncodingEqualsStdLengthEncoding(t *testing.T) {
	tests := [][]byte{
		{0x00},
		{0x00, 0x00, 0x01},
		{0x42, 0x43, 0x44},
		{0x00, 0x42, 0x00, 0x43},
	}

	for _, input := range tests {
		stdEncoded := StdEncoding.EncodeToString(input)
		stdLengthEncoded := StdLengthEncoding.EncodeToString(input)

		if stdEncoded != stdLengthEncoded {
			t.Errorf("StdEncoding and StdLengthEncoding produce different results: input=%v, std=%s, stdLength=%s",
				input, stdEncoded, stdLengthEncoded)
		}
	}
}

func TestFlipEncodingEqualsFlipLengthEncoding(t *testing.T) {
	tests := [][]byte{
		{0x00},
		{0x00, 0x00, 0x01},
		{0x42, 0x43, 0x44},
		{0x00, 0x42, 0x00, 0x43},
	}

	for _, input := range tests {
		flipEncoded := FlipEncoding.EncodeToString(input)
		flipLengthEncoded := FlipLengthEncoding.EncodeToString(input)

		if flipEncoded != flipLengthEncoded {
			t.Errorf("FlipEncoding and FlipLengthEncoding produce different results: input=%v, flip=%s, flipLength=%s",
				input, flipEncoded, flipLengthEncoded)
		}
	}
}

func TestAESUseCaseWithStdEncoding(t *testing.T) {
	aesKey := []byte{
		0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
		0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
	}

	encoded := StdEncoding.EncodeToString(aesKey)
	decoded := StdEncoding.DecodeString(encoded)

	if !bytes.Equal(decoded, aesKey) {
		t.Errorf("AES key encoding/decoding failed: original=%v, decoded=%v", aesKey, decoded)
	}

	if len(decoded) != len(aesKey) {
		t.Errorf("AES key length not preserved: expected=%d, got=%d", len(aesKey), len(decoded))
	}
}

func TestAESUseCaseWithFlipEncoding(t *testing.T) {
	aesKey := []byte{
		0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
		0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
		0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
		0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
	}

	encoded := FlipEncoding.EncodeToString(aesKey)
	decoded := FlipEncoding.DecodeString(encoded)

	if !bytes.Equal(decoded, aesKey) {
		t.Errorf("AES key encoding/decoding failed: original=%v, decoded=%v", aesKey, decoded)
	}

	if len(decoded) != len(aesKey) {
		t.Errorf("AES key length not preserved: expected=%d, got=%d", len(aesKey), len(decoded))
	}
}

func TestRandomDataRoundTrip(t *testing.T) {
	sizes := []int{1, 8, 16, 32, 64, 128, 256}

	for _, size := range sizes {
		t.Run("size_"+string(rune(size+'0')), func(t *testing.T) {
			data := make([]byte, size)
			_, err := rand.Read(data)
			if err != nil {
				t.Fatalf("Failed to generate random data: %v", err)
			}

			stdEncoded := StdEncoding.EncodeToString(data)
			stdDecoded := StdEncoding.DecodeString(stdEncoded)
			if !bytes.Equal(stdDecoded, data) {
				t.Errorf("StdEncoding round trip failed for size %d", size)
			}

			flipEncoded := FlipEncoding.EncodeToString(data)
			flipDecoded := FlipEncoding.DecodeString(flipEncoded)
			if !bytes.Equal(flipDecoded, data) {
				t.Errorf("FlipEncoding round trip failed for size %d", size)
			}
		})
	}
}

func TestEncodingDifference(t *testing.T) {
	input := []byte{0x42, 0x43, 0x44}

	stdEncoded := StdEncoding.EncodeToString(input)
	flipEncoded := FlipEncoding.EncodeToString(input)

	if stdEncoded == flipEncoded {
		t.Errorf("StdEncoding and FlipEncoding should produce different outputs, but both returned: %s", stdEncoded)
	}

	stdDecoded := StdEncoding.DecodeString(stdEncoded)
	flipDecoded := FlipEncoding.DecodeString(flipEncoded)

	if !bytes.Equal(stdDecoded, input) {
		t.Errorf("StdEncoding decode failed: expected=%v, got=%v", input, stdDecoded)
	}

	if !bytes.Equal(flipDecoded, input) {
		t.Errorf("FlipEncoding decode failed: expected=%v, got=%v", input, flipDecoded)
	}
}

func TestDecodeStringStrict(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		shouldErr bool
	}{
		{"valid std encoding", "ABC123", false},
		{"invalid character", "ABC@123", true},
		{"invalid character space", "ABC 123", true},
		{"empty string", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := StdEncoding.DecodeStringStrict(tt.input)
			if tt.shouldErr && err == nil {
				t.Errorf("Expected error for input %q, but got none", tt.input)
			}
			if !tt.shouldErr && err != nil {
				t.Errorf("Unexpected error for input %q: %v", tt.input, err)
			}
		})
	}
}

func BenchmarkStdEncodingEncode(b *testing.B) {
	data := make([]byte, 32)
	rand.Read(data)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = StdEncoding.EncodeToString(data)
	}
}

func BenchmarkStdEncodingDecode(b *testing.B) {
	data := make([]byte, 32)
	rand.Read(data)
	encoded := StdEncoding.EncodeToString(data)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = StdEncoding.DecodeString(encoded)
	}
}

func BenchmarkFlipEncodingEncode(b *testing.B) {
	data := make([]byte, 32)
	rand.Read(data)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = FlipEncoding.EncodeToString(data)
	}
}

func BenchmarkFlipEncodingDecode(b *testing.B) {
	data := make([]byte, 32)
	rand.Read(data)
	encoded := FlipEncoding.EncodeToString(data)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = FlipEncoding.DecodeString(encoded)
	}
}
