from typing import Union


class Data():
    def __init__(
        self, 
        name:          str   = "",
        price:         float = 0.0,
        pieces:        int   = 0,
        unique_pieces: int   = 0,
        theme:         str   = "",
        age_group:     int   = 0,
        manual_pages:  int   = 0,
        gender:        str   = ""
    ) -> None:
        self._name:          str   = name
        self._price:         float = price
        self._pieces:        int   = pieces
        self._unique_pieces: int   = unique_pieces
        self._theme:         str   = theme
        self._age_group:     int   = age_group
        self._manual_pages:  int   = manual_pages
        self._gender:        str   = gender
    
    def get_name(self) -> str:
        return self._name
        
    def get_price(self) -> float:
        return self._price
        
    def get_pieces(self) -> int:
        return self._pieces

    def get_unique_pieces(self) -> int:
        return self._unique_pieces
        
    def get_theme(self) -> str:
        return self._theme
        
    def get_age_group(self) -> int:
        return self._age_group
    
    def get_manual_pages(self) -> int:
        return self._manual_pages
        
    def get_gender(self) -> str:
        return self._gender
