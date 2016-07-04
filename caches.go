package hessfree

import (
	"sync"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

type deltaCache struct {
	vecs vecCache
}

func (d *deltaCache) Alloc(vars []*autofunc.Variable) ConstParamDelta {
	res := ConstParamDelta{}
	for _, v := range vars {
		res[v] = d.vecs.Alloc(len(v.Vector))
	}
	return res
}

func (d *deltaCache) Release(c ConstParamDelta) {
	for _, vec := range c {
		d.vecs.Release(vec)
	}
}

type vecCache struct {
	lock sync.Mutex
	vecs map[int][]linalg.Vector
}

func (v *vecCache) Alloc(size int) linalg.Vector {
	v.lock.Lock()
	if v.vecs == nil {
		v.vecs = map[int][]linalg.Vector{}
	}
	cache := v.vecs[size]
	if len(cache) > 0 {
		res := cache[len(cache)-1]
		cache[len(cache)-1] = nil
		v.vecs[size] = cache[:len(cache)-1]
		v.lock.Unlock()
		for i := range res {
			res[i] = 0
		}
		return res
	}
	v.lock.Unlock()
	return make(linalg.Vector, size)
}

func (v *vecCache) Release(vec linalg.Vector) {
	v.lock.Lock()
	defer v.lock.Unlock()
	if v.vecs == nil {
		v.vecs = map[int][]linalg.Vector{}
	}
	s := len(vec)
	v.vecs[s] = append(v.vecs[s], vec)
}
