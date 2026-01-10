use colored::{ColoredString, Colorize};

/// Trait for application-specific color themes.
///
/// This trait extends `colored::Colorize` to provide semantic color names
/// consistent with the application's design (Monokai-inspired).
pub trait ThemeColor: Colorize + Sized {
    fn primary(self) -> ColoredString {
        self.truecolor(100, 210, 255)
    }

    fn subtext(self) -> ColoredString {
        self.truecolor(100, 110, 150)
    }

    fn success(self) -> ColoredString {
        self.truecolor(80, 250, 210)
    }

    fn failure(self) -> ColoredString {
        self.truecolor(255, 90, 120)
    }

    fn info(self) -> ColoredString {
        self.truecolor(130, 170, 255)
    }

    fn bg_dark(self) -> ColoredString {
        self.truecolor(50, 55, 80)
    }

    fn text(self) -> ColoredString {
        self.truecolor(220, 230, 255)
    }

    fn warning(self) -> ColoredString {
        self.truecolor(255, 210, 100)
    }

    fn danger(self) -> ColoredString {
        self.truecolor(255, 160, 80)
    }
}

// Implement ThemeColor for all types that implement Colorize
impl<T: Colorize> ThemeColor for T {}
