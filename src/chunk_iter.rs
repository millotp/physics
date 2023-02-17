use std::marker::PhantomData;

pub struct ChunksMutIndices<'a, T: 'a> {
    v: *mut [T],
    breakpoints: &'a [usize],
    curr_ind: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T: 'a> ChunksMutIndices<'a, T> {
    #[inline]
    pub fn new(slice: &'a mut [T], breakpoints: &'a [usize]) -> Self {
        Self {
            v: slice,
            breakpoints,
            curr_ind: 0,
            _marker: PhantomData,
        }
    }
}

impl<'a, T> Iterator for ChunksMutIndices<'a, T> {
    type Item = (&'a mut [T], usize);

    #[inline]
    fn next(&mut self) -> Option<(&'a mut [T], usize)> {
        if self.v.is_empty() || self.curr_ind >= self.breakpoints.len() {
            None
        } else {
            let siz = if self.curr_ind < self.breakpoints.len() - 1 {
                self.breakpoints[self.curr_ind + 1] - self.breakpoints[self.curr_ind]
            } else {
                self.v.len()
            };
            self.curr_ind += 1;
            // SAFETY: The self.v contract ensures that any split_at_mut is valid.
            let (head, tail) = unsafe { self.v.split_at_mut(siz) };
            self.v = tail;
            // SAFETY: Nothing else points to or will point to the contents of this slice.
            Some(unsafe { (&mut *head, self.breakpoints[self.curr_ind - 1]) })
        }
    }
}
