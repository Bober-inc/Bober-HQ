class Data():
    """
    A class to handle and store toy data attributes
    """
    def __init__(self, data_list):
        """
        Initialize Data object with list of toy attributes

        Args:
            data_list (list): List containing toy data attributes
        """
        self.set_name(data_list[1])
        self.set_price(data_list[4:6])  # 4:Price, 5:Amazon Price
        self.set_pieces(data_list[3]) 
        self.set_unique_pieces(data_list[12])
        self.set_theme(data_list[2])
        self.set_age_group(data_list[7])
        self.set_manual_pages(data_list[8])
        self.set_gender(data_list[7])

    def set_name(self, name):
        """
        Set toy name

        Args:
            name: Name of the toy
        """
        if isinstance(name, str):
            self._name = name
        else:
            try:
                self._name = str(name)
            except:
                self._name = ""

    def set_price(self, price):
        """
        Set toy price

        Args:
            price (list): List containing regular and Amazon prices
        """
        try:
            self._price = float(price[0][1:])
        except:
            try:
                self._price = float(price[1][1:]) 
            except:
                self._price = -1

    def set_pieces(self, pieces):
        """
        Set number of pieces

        Args:
            pieces: Number of pieces in toy set
        """
        try:
            self._pieces = int(pieces)
        except:
            self._pieces = 0

    def set_unique_pieces(self, unique_pieces):
        """
        Set number of unique pieces

        Args:
            unique_pieces: Number of unique pieces in toy set
        """
        try:
            self._unique_pieces = int(unique_pieces)
        except:
            self._unique_pieces = 0

    def set_theme(self, theme):
        """
        Set toy theme

        Args:
            theme: Theme of the toy
        """
        if isinstance(theme, str):
            self._theme = theme
        else:
            try:
                self._theme = str(theme)
            except:
                self._theme = ""

    def set_age_group(self, age_group):
        """
        Set recommended age group

        Args:
            age_group (str): Age group in format 'Ages_x+'
        """
        try:
            age = age_group.replace("Ages_", "").replace("+", "")
            self._age_group = int(age)
        except:
            self._age_group = 0

    def set_manual_pages(self, manual_pages):
        """
        Set number of manual pages

        Args:
            manual_pages: Number of pages in instruction manual
        """
        try:
            self._manual_pages = int(manual_pages)
        except:
            self._manual_pages = 0

    def set_gender(self, gender):
        """
        Set target gender

        Args:
            gender: Target gender for toy
        """
        if isinstance(gender, str):
            self._gender = gender
        else:
            try:
                self._gender = str(gender)
            except:
                self._gender = ""

    def get_name(self):
        """
        Get toy name

        Returns:
            str: Name of the toy
        """
        return self._name

    def get_price(self):
        """
        Get toy price

        Returns:
            float: Price of the toy
        """
        return self._price

    def get_pieces(self):
        """
        Get number of pieces

        Returns:
            int: Number of pieces in toy set
        """
        return self._pieces

    def get_unique_pieces(self):
        """
        Get number of unique pieces

        Returns:
            int: Number of unique pieces in toy set
        """
        return self._unique_pieces

    def get_theme(self):
        """
        Get toy theme

        Returns:
            str: Theme of the toy
        """
        return self._theme

    """
    Get the age group as a number. The age group taken from 'Ages_x+' format

    Returns:
        int: appropriate age for toy
    """
    def get_age_group(self):
        return self._age_group

    def get_manual_pages(self):
        """
        Get number of manual pages

        Returns:
            int: Number of pages in instruction manual
        """
        return self._manual_pages

    def get_gender(self):
        """
        Get target gender

        Returns:
            str: Target gender for toy
        """
        return self._gender

    def get_all_fields(self):
        """
        Get string representation of all fields

        Returns:
            str: String containing all field values
        """
        return (f"Name: {self._name}\n"
                f"Price: ${self._price}\n"
                f"Pieces: {self._pieces}\n"
                f"Unique Pieces: {self._unique_pieces}\n"
                f"Theme: {self._theme}\n"
                f"Age Group: {self._age_group}+\n"
                f"Manual Pages: {self._manual_pages}\n"
                f"Gender: {self._gender}")
