pub struct ChunksMutIndices<'a, T: 'a> {
    v: Option<&'a mut [T]>,
    breakpoints: &'a [usize],
    curr_ind: usize,
}

impl<'a, T: 'a + Sync> ChunksMutIndices<'a, T> {
    #[inline]
    pub fn new(slice: &'a mut [T], breakpoints: &'a [usize]) -> Self {
        Self {
            v: Some(slice),
            breakpoints,
            curr_ind: 0,
        }
    }
}

impl<'a, T> Iterator for ChunksMutIndices<'a, T> {
    type Item = (&'a mut [T], usize);

    #[inline]
    fn next(&mut self) -> Option<(&'a mut [T], usize)> {
        if self.curr_ind >= self.breakpoints.len() {
            None
        } else {
            let v = self.v.take()?;
            let siz = if self.curr_ind < self.breakpoints.len() - 1 {
                self.breakpoints[self.curr_ind + 1] - self.breakpoints[self.curr_ind]
            } else {
                v.len()
            };
            let (head, tail) = v.split_at_mut(siz);
            self.v = Some(tail);
            self.curr_ind += 1;
            Some((head, self.breakpoints[self.curr_ind - 1]))
        }
    }
}
