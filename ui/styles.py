
class ModernTheme:
    """
    Defines modern, professional QSS stylesheets for the application.
    """
    
    FONT_FAMILY = "Segoe UI, 'Malgun Gothic', sans-serif"
    FONT_SIZE_MAIN = "10pt"
    
    # --- Dark Theme Palette ---
    DARK_BG_MAIN = "#202124"        # Google/VSCode Dark
    DARK_BG_PANEL = "#2d2e30"       # Panels/Containers
    DARK_BG_INPUT = "#37373d"       # Inputs
    DARK_BORDER = "#4a4a4f"
    DARK_TEXT_MAIN = "#e8eaed"
    DARK_TEXT_SUB = "#9aa0a6"
    DARK_ACCENT = "#8ab4f8"         # Gentle Blue
    DARK_ACCENT_HOVER = "#alc9fc"
    DARK_BTN_BG = "#3c4043"
    DARK_BTN_HOVER = "#4a4e51"
    DARK_SCROLL_HANDLE = "#5f6368"
    
    # --- Light Theme Palette ---
    LIGHT_BG_MAIN = "#f8f9fa"       # Google Light
    LIGHT_BG_PANEL = "#ffffff"
    LIGHT_BG_INPUT = "#ffffff"
    LIGHT_BORDER = "#dadce0"
    LIGHT_TEXT_MAIN = "#202124"
    LIGHT_TEXT_SUB = "#5f6368"
    LIGHT_ACCENT = "#1a73e8"        # Google Blue
    LIGHT_ACCENT_HOVER = "#1557b0"
    LIGHT_BTN_BG = "#f1f3f4"
    LIGHT_BTN_HOVER = "#e8eaed"
    LIGHT_SCROLL_HANDLE = "#dadce0"

    @staticmethod
    def get_dark_theme():
        return f"""
            /* Global Reset */
            * {{
                font-family: {ModernTheme.FONT_FAMILY};
                font-size: {ModernTheme.FONT_SIZE_MAIN};
                color: {ModernTheme.DARK_TEXT_MAIN};
            }}
            
            QMainWindow, QWidget {{
                background-color: {ModernTheme.DARK_BG_MAIN};
                outline: none;
            }}
            
            /* Panels & Containers */
            QGroupBox {{
                background-color: {ModernTheme.DARK_BG_PANEL};
                border: 1px solid {ModernTheme.DARK_BORDER};
                border-radius: 8px;
                margin-top: 1.2em; /* Reduced from 1.5em */
                padding: 6px; /* Reduced from 10px */
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                left: 10px;
                color: {ModernTheme.DARK_ACCENT};
                font-weight: bold;
                font-size: 10pt; /* Slightly smaller title */
            }}
            
            /* Tabs */
            QTabWidget::pane {{
                border: 1px solid {ModernTheme.DARK_BORDER};
                background-color: {ModernTheme.DARK_BG_PANEL};
                border-radius: 4px;
            }}
            QTabBar::tab {{
                background: {ModernTheme.DARK_BG_MAIN};
                color: {ModernTheme.DARK_TEXT_SUB};
                padding: 8px 16px;
                border: 1px solid transparent;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 80px;
            }}
            QTabBar::tab:selected {{
                background: {ModernTheme.DARK_BG_PANEL};
                color: {ModernTheme.DARK_ACCENT};
                border-bottom: 2px solid {ModernTheme.DARK_ACCENT};
                font-weight: bold;
            }}
            QTabBar::tab:hover:!selected {{
                background: {ModernTheme.DARK_BTN_BG};
                color: {ModernTheme.DARK_TEXT_MAIN};
            }}
            
            /* Inputs */
            QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                background-color: {ModernTheme.DARK_BG_INPUT};
                border: 1px solid {ModernTheme.DARK_BORDER};
                border-radius: 6px; # Rounded inputs
                padding: 4px 8px;
                selection-background-color: {ModernTheme.DARK_ACCENT};
                selection-color: {ModernTheme.DARK_BG_MAIN};
            }}
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {{
                border: 1px solid {ModernTheme.DARK_ACCENT};
            }}
            
            /* Buttons */
            QPushButton {{
                background-color: {ModernTheme.DARK_BTN_BG};
                border: 1px solid {ModernTheme.DARK_BORDER};
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {ModernTheme.DARK_BTN_HOVER};
                border-color: {ModernTheme.DARK_TEXT_SUB};
            }}
            QPushButton:pressed {{
                background-color: {ModernTheme.DARK_ACCENT};
                color: {ModernTheme.DARK_BG_MAIN};
            }}
            QPushButton:disabled {{
                background-color: {ModernTheme.DARK_BG_MAIN};
                color: {ModernTheme.DARK_BORDER};
                border: 1px solid {ModernTheme.DARK_BORDER};
            }}
            
            /* ComboBox details */
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 0px;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid {ModernTheme.DARK_TEXT_SUB};
                margin-right: 5px;
            }}
            
            /* ScrollBar */
            QScrollBar:vertical {{
                border: none;
                background: {ModernTheme.DARK_BG_MAIN};
                width: 10px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {ModernTheme.DARK_SCROLL_HANDLE};
                min-height: 20px;
                border-radius: 5px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
             QScrollBar:horizontal {{
                border: none;
                background: {ModernTheme.DARK_BG_MAIN};
                height: 10px;
                margin: 0px;
            }}
            QScrollBar::handle:horizontal {{
                background: {ModernTheme.DARK_SCROLL_HANDLE};
                min-width: 20px;
                border-radius: 5px;
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
            
            /* Splitter */
            QSplitter::handle {{
                background-color: {ModernTheme.DARK_BORDER};
                width: 2px;
            }}
            
            /* Menus */
            QMenuBar {{
                background-color: {ModernTheme.DARK_BG_MAIN};
                border-bottom: 1px solid {ModernTheme.DARK_BORDER};
            }}
            QMenuBar::item {{
                padding: 5px 10px;
                background: transparent;
            }}
            QMenuBar::item:selected {{
                background: {ModernTheme.DARK_BTN_BG};
            }}
            QMenu {{
                background-color: {ModernTheme.DARK_BG_PANEL};
                border: 1px solid {ModernTheme.DARK_BORDER};
            }}
            QMenu::item {{
                padding: 5px 20px;
            }}
            QMenu::item:selected {{
                background-color: {ModernTheme.DARK_BTN_HOVER};
            }}
            
            /* Specific Highlights */
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid {ModernTheme.DARK_TEXT_SUB};
                border-radius: 3px;
                background: {ModernTheme.DARK_BG_INPUT};
            }}
            QCheckBox::indicator:checked {{
                background-color: {ModernTheme.DARK_ACCENT};
                border-color: {ModernTheme.DARK_ACCENT};
            }}
            
            QRadioButton::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid {ModernTheme.DARK_TEXT_SUB};
                border-radius: 8px;
                background: {ModernTheme.DARK_BG_INPUT};
            }}
            QRadioButton::indicator:checked {{
                background-color: {ModernTheme.DARK_ACCENT};
                border-color: {ModernTheme.DARK_ACCENT};
            }}
            
            /* Tooltips */
            /* Tooltips */
            QToolTip {{
                color: #ffffff;
                background-color: #333333;
                border: 1px solid #777;
                padding: 5px;
            }}

            /* Specific IDs & Classes */
            /* Clean SpinBox (Hidden Arrows) */
            QSpinBox#clean_spin, QDoubleSpinBox#clean_spin {{
                border: 1px solid {ModernTheme.DARK_BORDER};
                border-radius: 4px;
                background-color: {ModernTheme.DARK_BG_INPUT};
                padding-left: 0px;
                padding-right: 0px;
            }}
            QSpinBox#clean_spin::up-button, QSpinBox#clean_spin::down-button,
            QDoubleSpinBox#clean_spin::up-button, QDoubleSpinBox#clean_spin::down-button {{
                width: 0px; 
                height: 0px;
                border: none;
            }} 
            
            QTextEdit#pos_prompt {{ 
                border: 2px solid #4dabf7; 
                background-color: #263238; 
            }}
            QTextEdit#neg_prompt {{ 
                border: 2px solid #e74c3c; 
                background-color: #3e2723; 
            }}
            QTextEdit#yolo_classes {{ 
                border: 1px solid #9b59b6; 
            }}
            QCheckBox#important_chk {{
                font-weight: bold;
                color: #4dabf7; /* Blue accent */
            }}
            QCheckBox#purple_chk {{
                font-weight: bold;
                color: #af7ac5; /* Purple */
            }}
            QPushButton#warning_btn {{
                background-color: #e74c3c; 
                color: white; 
                border: 1px solid #c0392b;
            }}
            QPushButton#warning_btn:hover {{
                background-color: #c0392b;
            }}
        """

    @staticmethod
    def get_light_theme():
        return f"""
            /* Global Reset */
            * {{
                font-family: {ModernTheme.FONT_FAMILY};
                font-size: {ModernTheme.FONT_SIZE_MAIN};
                color: {ModernTheme.LIGHT_TEXT_MAIN};
            }}
            
            QMainWindow, QWidget {{
                background-color: {ModernTheme.LIGHT_BG_MAIN};
                outline: none;
            }}
            
            /* Panels & Containers */
            QGroupBox {{
                background-color: {ModernTheme.LIGHT_BG_PANEL};
                border: 1px solid {ModernTheme.LIGHT_BORDER};
                border-radius: 8px;
                margin-top: 1.2em; 
                padding: 6px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                left: 10px;
                color: {ModernTheme.LIGHT_ACCENT};
                font-weight: bold;
                font-size: 10pt;
            }}
            
            /* Tabs */
            QTabWidget::pane {{
                border: 1px solid {ModernTheme.LIGHT_BORDER};
                background-color: {ModernTheme.LIGHT_BG_PANEL};
                border-radius: 4px;
            }}
            QTabBar::tab {{
                background: {ModernTheme.LIGHT_BG_MAIN};
                color: {ModernTheme.LIGHT_TEXT_SUB};
                padding: 8px 16px;
                border: 1px solid transparent;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 80px;
            }}
            QTabBar::tab:selected {{
                background: {ModernTheme.LIGHT_BG_PANEL};
                color: {ModernTheme.LIGHT_ACCENT};
                border-bottom: 2px solid {ModernTheme.LIGHT_ACCENT};
                font-weight: bold;
            }}
            QTabBar::tab:hover:!selected {{
                background: {ModernTheme.LIGHT_BTN_BG};
                color: {ModernTheme.LIGHT_TEXT_MAIN};
            }}
            
            /* Inputs */
            QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                background-color: {ModernTheme.LIGHT_BG_INPUT};
                border: 1px solid {ModernTheme.LIGHT_BORDER};
                border-radius: 6px; 
                padding: 4px 8px;
                selection-background-color: {ModernTheme.LIGHT_ACCENT};
                selection-color: {ModernTheme.LIGHT_BG_PANEL};
            }}
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {{
                border: 1px solid {ModernTheme.LIGHT_ACCENT};
            }}
            
            /* Buttons */
            QPushButton {{
                background-color: {ModernTheme.LIGHT_BTN_BG};
                border: 1px solid {ModernTheme.LIGHT_BORDER};
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {ModernTheme.LIGHT_BTN_HOVER};
                border-color: {ModernTheme.LIGHT_TEXT_SUB};
            }}
            QPushButton:pressed {{
                background-color: {ModernTheme.LIGHT_ACCENT};
                color: {ModernTheme.LIGHT_BG_PANEL};
            }}
            QPushButton:disabled {{
                background-color: {ModernTheme.LIGHT_BG_MAIN};
                color: {ModernTheme.LIGHT_BORDER};
                border: 1px solid {ModernTheme.LIGHT_BORDER};
            }}
            
            /* ComboBox details */
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 0px;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid {ModernTheme.LIGHT_TEXT_SUB};
                margin-right: 5px;
            }}
            
            /* ScrollBar */
            QScrollBar:vertical {{
                border: none;
                background: {ModernTheme.LIGHT_BG_MAIN};
                width: 10px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {ModernTheme.LIGHT_SCROLL_HANDLE};
                min-height: 20px;
                border-radius: 5px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
             QScrollBar:horizontal {{
                border: none;
                background: {ModernTheme.LIGHT_BG_MAIN};
                height: 10px;
                margin: 0px;
            }}
            QScrollBar::handle:horizontal {{
                background: {ModernTheme.LIGHT_SCROLL_HANDLE};
                min-width: 20px;
                border-radius: 5px;
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
            
            /* Splitter */
            QSplitter::handle {{
                background-color: {ModernTheme.LIGHT_BORDER};
                width: 2px;
            }}
            
            /* Menus */
            QMenuBar {{
                background-color: {ModernTheme.LIGHT_BG_MAIN};
                border-bottom: 1px solid {ModernTheme.LIGHT_BORDER};
            }}
            QMenuBar::item {{
                padding: 5px 10px;
                background: transparent;
            }}
            QMenuBar::item:selected {{
                background: {ModernTheme.LIGHT_BTN_BG};
            }}
            QMenu {{
                background-color: {ModernTheme.LIGHT_BG_PANEL};
                border: 1px solid {ModernTheme.LIGHT_BORDER};
            }}
            QMenu::item {{
                padding: 5px 20px;
            }}
            QMenu::item:selected {{
                background-color: {ModernTheme.LIGHT_BTN_HOVER};
            }}

            /* Specific Highlights */
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid {ModernTheme.LIGHT_TEXT_SUB};
                border-radius: 3px;
                background: {ModernTheme.LIGHT_BG_INPUT};
            }}
            QCheckBox::indicator:checked {{
                background-color: {ModernTheme.LIGHT_ACCENT};
                border-color: {ModernTheme.LIGHT_ACCENT};
            }}
            
            QRadioButton::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid {ModernTheme.LIGHT_TEXT_SUB};
                border-radius: 8px;
                background: {ModernTheme.LIGHT_BG_INPUT};
            }}
            QRadioButton::indicator:checked {{
                background-color: {ModernTheme.LIGHT_ACCENT};
                border-color: {ModernTheme.LIGHT_ACCENT};
            }}
            
            /* Tooltips */
            /* Tooltips */
            QToolTip {{
                color: #333;
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                padding: 5px;
            }}

            /* Specific IDs & Classes */
            /* Clean SpinBox (Hidden Arrows) */
            QSpinBox#clean_spin, QDoubleSpinBox#clean_spin {{
                border: 1px solid {ModernTheme.LIGHT_BORDER};
                border-radius: 4px;
                background-color: {ModernTheme.LIGHT_BG_INPUT};
                padding-left: 0px;
                padding-right: 0px;
            }}
            QSpinBox#clean_spin::up-button, QSpinBox#clean_spin::down-button,
            QDoubleSpinBox#clean_spin::up-button, QDoubleSpinBox#clean_spin::down-button {{
                width: 0px; 
                height: 0px;
                border: none;
            }} 
            
            QTextEdit#pos_prompt {{ 
                border: 2px solid #2980b9; 
                background-color: #eaf2f8; 
            }}
            QTextEdit#neg_prompt {{ 
                border: 2px solid #c0392b; 
                background-color: #f9ebea; 
            }}
            QTextEdit#yolo_classes {{ 
                border: 1px solid #8e44ad; 
            }}
            QCheckBox#important_chk {{
                font-weight: bold;
                color: #1a73e8; /* Blue accent */
            }}
            QCheckBox#purple_chk {{
                font-weight: bold;
                color: #8e44ad; /* Purple */
            }}
            QPushButton#warning_btn {{
                background-color: #e74c3c; 
                color: white; 
                border: 1px solid #c0392b;
            }}
            QPushButton#warning_btn:hover {{
                background-color: #c0392b;
            }}
        """
