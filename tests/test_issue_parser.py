import unittest
from github_issue_creator import parse_development_plan, parse_recommendations

DEV_MD = """
### Phase 2 (Example)

- Do thing A *
- Completed item ✓
- Failing item !

### Documentation Updates Needed
- Add step to README
- Clarify version policy ✓
"""

REC_MD = """
#### Priority 1 - Immediate Value

1. **Personal Knowledge Graph**
   - Build clusters
   - Search utility

2. **Learning Gap Analysis**
   - Identify gaps

#### Priority 2 - Advanced Analytics

1. **Timeline**
   - Monthly aggregates
"""


class TestIssueParser(unittest.TestCase):
    def test_parse_development_plan(self):
        issues = parse_development_plan(DEV_MD, "development_plan.md")
        titles = [i.title for i in issues]
        # Completed item should be excluded
        self.assertTrue(any("Do thing A" in t for t in titles))
        self.assertFalse(any("Completed item" in t for t in titles))
        # Docs item should be included
        self.assertTrue(any(t.startswith("[Docs]") for t in titles))

    def test_parse_recommendations(self):
        issues = parse_recommendations(REC_MD, "data-mining-recommendations.md")
        titles = [i.title for i in issues]
        self.assertTrue(any("[Priority 1] Personal Knowledge Graph" == t for t in titles))
        self.assertTrue(any("[Priority 2] Timeline" == t for t in titles))


if __name__ == "__main__":
    unittest.main()
