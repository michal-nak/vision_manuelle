# Contributing Guide

## Welcome!

Thank you for considering contributing to Gesture Paint! This document provides guidelines for contributing to the project.

## Getting Started

### Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/vision_manuelle.git
   cd vision_manuelle
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python main.py
   ```

### Project Structure

Please review `docs/ARCHITECTURE.md` to understand the codebase organization before making changes.

## How to Contribute

### Reporting Bugs

**Before submitting a bug report**:
- Check existing issues to avoid duplicates
- Test with the latest version
- Gather relevant information (OS, Python version, error messages)

**Bug Report Template**:
```markdown
**Description**: Brief description of the bug

**Steps to Reproduce**:
1. Launch application with...
2. Make gesture...
3. Observe error...

**Expected Behavior**: What should happen

**Actual Behavior**: What actually happens

**Environment**:
- OS: Windows/Linux/macOS
- Python version: 3.x.x
- Detection mode: MediaPipe/CV
- Camera: Built-in/USB webcam

**Error Messages**: (paste console output)

**Screenshots**: (if applicable)
```

### Suggesting Features

**Feature Request Template**:
```markdown
**Feature Description**: Clear description of the feature

**Use Case**: Why is this feature needed?

**Proposed Solution**: How might this work?

**Alternatives**: Other approaches considered

**Additional Context**: Any other relevant information
```

### Submitting Pull Requests

#### PR Checklist
- [ ] Code follows project style guidelines
- [ ] Changes are tested and working
- [ ] Documentation is updated if needed
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains what and why

#### PR Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clean, readable code
   - Add comments for complex logic
   - Follow existing patterns

3. **Test thoroughly**:
   - Test with both MediaPipe and CV modes
   - Verify on your platform
   - Check for regressions

4. **Commit changes**:
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**:
   - Go to original repository
   - Click "New Pull Request"
   - Select your branch
   - Fill in PR template

#### PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- Tested on: [OS/Platform]
- Detection modes tested: MediaPipe / CV / Both
- Test scenarios: [describe what you tested]

## Screenshots
(if applicable)

## Related Issues
Fixes #[issue number]
```

## Coding Standards

### Python Style

Follow PEP 8 with these project-specific conventions:

**Naming**:
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_leading_underscore`

**Docstrings**:
```python
def function_name(param1, param2):
    """
    Brief description of function
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
    """
    pass
```

**Imports**:
- Standard library first
- Third-party libraries second
- Local modules last
- Alphabetical within groups

### Code Organization

**Modularity**:
- Keep functions focused and single-purpose
- Extract reusable logic into separate modules
- Use inheritance for shared behavior

**Error Handling**:
```python
try:
    # Risky operation
    result = perform_operation()
except SpecificException as e:
    # Handle specific case
    logger.error(f"Operation failed: {e}")
    return fallback_value
```

**Configuration**:
- Use `config.py` for shared constants
- Load from JSON files when appropriate
- Provide sensible defaults

### Testing

**Manual Testing**:
- Test with both detection modes
- Verify all gestures work
- Check edge cases (no hand, multiple hands, poor lighting)
- Test on your platform

**Add Test Scenarios**:
When adding features, document test scenarios:
```python
# Test: Gesture recognition with poor lighting
# Expected: Should still detect hand with default config
# Actual: [your results]
```

## Documentation

### When to Update Docs

**Always**:
- Adding new features → Update `README.md` and `docs/USAGE.md`
- Changing architecture → Update `docs/ARCHITECTURE.md`
- Adding gestures → Update gesture documentation in UI and docs
- Fixing bugs → Update changelog

**Code Comments**:
- Complex algorithms need explanation
- Non-obvious design decisions should be documented
- Parameter ranges and units should be specified

### Documentation Style

- Use clear, concise language
- Include code examples
- Add screenshots for UI changes
- Update table of contents if needed

## Areas for Contribution

### Beginner-Friendly
- Documentation improvements
- UI enhancements
- Bug fixes
- Additional gestures
- Color themes

### Intermediate
- Performance optimizations
- New gesture recognition algorithms
- Multi-hand support
- Additional drawing tools (shapes, fill)
- Platform-specific improvements

### Advanced
- 3D hand tracking
- Machine learning gesture learning
- Remote collaboration features
- Plugin system
- Advanced image processing

## Code Review Process

### What We Look For

**Functionality**:
- Does it work as intended?
- Are edge cases handled?
- Is error handling appropriate?

**Code Quality**:
- Is it readable and maintainable?
- Does it follow project conventions?
- Is it well-organized?

**Testing**:
- Has it been tested?
- Are test results documented?

**Documentation**:
- Are changes documented?
- Are docstrings updated?
- Is user-facing documentation current?

### Feedback

- Be respectful and constructive
- Explain the "why" behind suggestions
- Acknowledge good solutions
- Be open to discussion

## Development Tips

### Debugging

**Enable Debug Mode**:
```python
# In gesture_paint.py, set default:
self.debug_var = tk.BooleanVar(value=True)
```

**Use Visualization Tools**:
```bash
# Interactive skin detection tuning
python tools/skin_tuner.py

# Pipeline step-by-step view
python tools/debug_detection.py
```

**Console Logging**:
```python
print(f"[DEBUG] Variable value: {value}")
```

### Common Pitfalls

**Camera Access**:
- Ensure no other app is using camera
- Use correct backend for your platform
- Check camera permissions

**Threading Issues**:
- Use `root.after()` for UI updates from threads
- Don't block the main thread
- Be careful with shared state

**Performance**:
- Avoid unnecessary processing in loop
- Use efficient NumPy operations
- Profile before optimizing

### Useful Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [Tkinter Reference](https://docs.python.org/3/library/tkinter.html)
- [NumPy Documentation](https://numpy.org/doc/)

## Communication

### Where to Ask Questions

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Pull Request Comments**: Code-specific discussions

### Response Times

- We aim to respond to issues within 3-5 days
- PRs may take 1-2 weeks for review
- Complex features may require multiple review cycles

## Recognition

Contributors will be:
- Listed in project README
- Credited in release notes
- Appreciated in commit messages

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

## Thank You!

Your contributions make this project better for everyone. Whether you're fixing a typo or implementing a major feature, every contribution is valued.

Happy coding! 🎨✋
