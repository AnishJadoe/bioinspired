        for token in self.tokens:
            self._build_token(token)
        for moveable_tile in self.movable_tiles:
            self._build_move_tiles(moveable_tile)