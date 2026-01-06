"""
Guideline synthesis: Use LLM to synthesize meta-guidelines from drill clusters.

Uses gpt-5.2 with extended thinking to analyze clusters of hard drills
and extract common failure patterns and strategic principles.
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import litellm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class GuidelineSynthesizer:
    """
    Synthesize meta-guidelines from clusters of hard drills using LLM.
    
    Strategy:
    1. For each cluster, identify drills with human guidelines
    2. Extract the common failure pattern from the guidelines
    3. Generalize to cover all drills in the cluster
    """
    
    def __init__(self, model: str = "openai/gpt-5.2", reasoning_effort: str = "high"):
        """
        Initialize synthesizer.
        
        Args:
            model: LLM model to use (litellm format)
            reasoning_effort: Reasoning effort for gpt-5.2 ("low", "medium", "high")
        """
        self.model = model
        self.reasoning_effort = reasoning_effort
    
    def _format_drill_brief(self, drill: Dict[str, Any]) -> str:
        """Format a drill as a brief description."""
        drill_id = drill['drill_id']
        name = drill['name']
        action_type = drill['expected_action'].get('type', 'unknown')
        has_guideline = '★' if drill.get('guideline_text') else ''
        
        return f"  Drill #{drill_id} {has_guideline}: {name} (action: {action_type})"
    
    def _format_drill_with_guideline(self, drill: Dict[str, Any]) -> str:
        """Format a drill WITH its human guideline."""
        brief = self._format_drill_brief(drill)
        guideline = drill.get('guideline_text', 'NO GUIDELINE')
        
        # Truncate guideline if too long
        if len(guideline) > 300:
            guideline = guideline[:300] + "..."
        
        return f"{brief}\n    Guideline: {guideline}"
    
    def synthesize_meta_guideline(
        self,
        cluster_drills: List[Dict[str, Any]],
        cluster_id: int
    ) -> Dict[str, Any]:
        """
        Synthesize a meta-guideline for a cluster of drills.
        
        Args:
            cluster_drills: List of drill dictionaries in this cluster
            cluster_id: Cluster ID number
            
        Returns:
            Dictionary with meta_guideline, failure_pattern, and metadata
        """
        # Separate drills with/without guidelines
        with_guidelines = [d for d in cluster_drills if d.get('guideline_text')]
        without_guidelines = [d for d in cluster_drills if not d.get('guideline_text')]
        
        print(f"\nCluster {cluster_id}: {len(cluster_drills)} drills ({len(with_guidelines)} with guidelines)", flush=True)
        
        # Build prompt
        prompt = self._build_synthesis_prompt(
            cluster_id,
            with_guidelines,
            without_guidelines
        )
        
        # Call LLM with thinking
        try:
            response = litellm.completion(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                reasoning_effort=self.reasoning_effort if 'gpt-5.2' in self.model else None
            )
            
            meta_guideline = response.choices[0].message.content
            
            print(f"  Synthesized {len(meta_guideline)} chars", flush=True)
            
        except Exception as e:
            print(f"  Error synthesizing guideline: {e}", flush=True)
            meta_guideline = f"[Synthesis failed for cluster {cluster_id}]"
        
        # Extract failure pattern (common themes in guidelines)
        failure_pattern = self._extract_failure_pattern(with_guidelines)
        
        return {
            'cluster_id': cluster_id,
            'meta_guideline': meta_guideline,
            'failure_pattern': failure_pattern,
            'num_drills': len(cluster_drills),
            'num_with_guidelines': len(with_guidelines),
            'drill_ids': [d['drill_id'] for d in cluster_drills],
            'action_types': list(set(d['expected_action'].get('type') for d in cluster_drills))
        }
    
    def _build_synthesis_prompt(
        self,
        cluster_id: int,
        with_guidelines: List[Dict[str, Any]],
        without_guidelines: List[Dict[str, Any]]
    ) -> str:
        """Build the LLM prompt for guideline synthesis."""
        
        total = len(with_guidelines) + len(without_guidelines)
        
        prompt = f"""You are analyzing a cluster of {total} difficult Catan drill scenarios where an LLM struggles to make the correct decision.

{len(with_guidelines)} drills in this cluster have HUMAN-WRITTEN GUIDELINES (marked with ★) that were created specifically to help the LLM avoid common mistakes. These guidelines reveal what the LLM typically gets wrong in these situations.

Your task: Synthesize a single META-GUIDELINE that:
1. Identifies the COMMON FAILURE PATTERN across these drills (why LLMs struggle here)
2. Extracts the shared strategic principle from the human guidelines
3. Generalizes to cover ALL drills in this cluster, including those without individual guidelines
4. Provides specific, actionable advice that an LLM can follow

DRILLS WITH HUMAN GUIDELINES (these reveal the failure pattern):
{self._format_drills_section(with_guidelines[:10])}  # Limit to first 10 for context
"""
        
        if len(with_guidelines) > 10:
            prompt += f"\n  ... and {len(with_guidelines) - 10} more drills with guidelines\n"
        
        if without_guidelines:
            prompt += f"""
DRILLS WITHOUT GUIDELINES (need help on similar situations):
{self._format_drills_section(without_guidelines[:5])}  # Sample
"""
            if len(without_guidelines) > 5:
                prompt += f"  ... and {len(without_guidelines) - 5} more drills\n"
        
        prompt += """

OUTPUT FORMAT:
Write a concise meta-guideline (2-4 sentences) that:
- Starts with the SITUATION (when does this apply?)
- Describes the FAILURE MODE (what LLMs typically get wrong)
- Provides the CORRECT APPROACH (what to do instead)
- Is specific enough to be actionable but general enough to cover all drills in this cluster

Example format:
"When [situation], LLMs often [failure mode] because [reason]. Instead, [correct approach]. [Additional specific advice if needed]."

Focus on extracting the INSIGHT behind the human guidelines, not just restating them."""
        
        return prompt
    
    def _format_drills_section(self, drills: List[Dict[str, Any]]) -> str:
        """Format a list of drills with their guidelines."""
        lines = []
        for drill in drills:
            if drill.get('guideline_text'):
                lines.append(self._format_drill_with_guideline(drill))
            else:
                lines.append(self._format_drill_brief(drill))
        return '\n'.join(lines)
    
    def _extract_failure_pattern(self, with_guidelines: List[Dict[str, Any]]) -> str:
        """
        Extract common failure pattern from guidelines.
        
        This is a simple keyword extraction approach.
        """
        if not with_guidelines:
            return "unknown"
        
        # Common failure pattern keywords
        keywords = {
            'setup': ['setup', 'placement', 'initial'],
            'road_building': ['road', 'extend', 'reach', 'path'],
            'resource_management': ['conserve', 'resource', 'trade', 'excess'],
            'trade_evaluation': ['trade', 'accept', 'reject', 'offer'],
            'timing': ['wait', 'end turn', 'hold', 'conserve'],
            'strategic_priority': ['priority', 'prefer', 'best', 'optimal'],
        }
        
        # Count keyword occurrences in guidelines
        pattern_scores = {pattern: 0 for pattern in keywords}
        for drill in with_guidelines:
            guideline = drill.get('guideline_text', '').lower()
            for pattern, kws in keywords.items():
                if any(kw in guideline for kw in kws):
                    pattern_scores[pattern] += 1
        
        # Return most common pattern
        if max(pattern_scores.values()) > 0:
            return max(pattern_scores, key=pattern_scores.get)
        return "general_decision_making"

