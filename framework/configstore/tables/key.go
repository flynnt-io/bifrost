package tables

import (
	"encoding/json"
	"time"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/schemas"
	"gorm.io/gorm"
)

// TableKey represents an API key configuration in the database
type TableKey struct {
	ID         uint      `gorm:"primaryKey;autoIncrement" json:"id"`
	Name       string    `gorm:"type:varchar(255);uniqueIndex:idx_key_name;not null" json:"name"`
	ProviderID uint      `gorm:"index;not null" json:"provider_id"`
	Provider   string    `gorm:"index;type:varchar(50)" json:"provider"`                          // ModelProvider as string
	KeyID      string    `gorm:"type:varchar(255);uniqueIndex:idx_key_id;not null" json:"key_id"` // UUID from schemas.Key
	Value      string    `gorm:"type:text;not null" json:"value"`
	ModelsJSON string    `gorm:"type:text" json:"-"` // JSON serialized []string
	Weight     float64   `gorm:"default:1.0" json:"weight"`
	CreatedAt  time.Time `gorm:"index;not null" json:"created_at"`
	UpdatedAt  time.Time `gorm:"index;not null" json:"updated_at"`

	// Azure config fields (embedded instead of separate table for simplicity)
	AzureEndpoint        *string `gorm:"type:text" json:"azure_endpoint,omitempty"`
	AzureAPIVersion      *string `gorm:"type:varchar(50)" json:"azure_api_version,omitempty"`
	AzureDeploymentsJSON *string `gorm:"type:text" json:"-"` // JSON serialized map[string]string

	// Vertex config fields (embedded)
	VertexProjectID       *string `gorm:"type:varchar(255)" json:"vertex_project_id,omitempty"`
	VertexRegion          *string `gorm:"type:varchar(100)" json:"vertex_region,omitempty"`
	VertexAuthCredentials *string `gorm:"type:text" json:"vertex_auth_credentials,omitempty"`

	// Bedrock config fields (embedded)
	BedrockAccessKey       *string `gorm:"type:varchar(255)" json:"bedrock_access_key,omitempty"`
	BedrockSecretKey       *string `gorm:"type:text" json:"bedrock_secret_key,omitempty"`
	BedrockSessionToken    *string `gorm:"type:text" json:"bedrock_session_token,omitempty"`
	BedrockRegion          *string `gorm:"type:varchar(100)" json:"bedrock_region,omitempty"`
	BedrockARN             *string `gorm:"type:text" json:"bedrock_arn,omitempty"`
	BedrockDeploymentsJSON *string `gorm:"type:text" json:"-"` // JSON serialized map[string]string

	// Apertus config fields (embedded)
	ApertusEndpoint *string `gorm:"type:text" json:"apertus_endpoint,omitempty"`

	// Virtual fields for runtime use (not stored in DB)
	Models            []string                   `gorm:"-" json:"models"`
	AzureKeyConfig    *schemas.AzureKeyConfig    `gorm:"-" json:"azure_key_config,omitempty"`
	VertexKeyConfig   *schemas.VertexKeyConfig   `gorm:"-" json:"vertex_key_config,omitempty"`
	BedrockKeyConfig  *schemas.BedrockKeyConfig  `gorm:"-" json:"bedrock_key_config,omitempty"`
	ApertusKeyConfig  *schemas.ApertusKeyConfig  `gorm:"-" json:"apertus_key_config,omitempty"`
}

// TableName sets the table name for each model
func (TableKey) TableName() string { return "config_keys" }

func (k *TableKey) BeforeSave(tx *gorm.DB) error {

	if k.Models != nil {
		data, err := json.Marshal(k.Models)
		if err != nil {
			return err
		}
		k.ModelsJSON = string(data)
	} else {
		k.ModelsJSON = "[]"
	}

	if k.AzureKeyConfig != nil {
		if k.AzureKeyConfig.Endpoint != "" {
			k.AzureEndpoint = &k.AzureKeyConfig.Endpoint
		} else {
			k.AzureEndpoint = nil
		}
		k.AzureAPIVersion = k.AzureKeyConfig.APIVersion
		if k.AzureKeyConfig.Deployments != nil {
			data, err := json.Marshal(k.AzureKeyConfig.Deployments)
			if err != nil {
				return err
			}
			s := string(data)
			k.AzureDeploymentsJSON = &s
		} else {
			k.AzureDeploymentsJSON = nil
		}
	} else {
		k.AzureEndpoint = nil
		k.AzureAPIVersion = nil
		k.AzureDeploymentsJSON = nil
	}

	if k.VertexKeyConfig != nil {
		if k.VertexKeyConfig.ProjectID != "" {
			k.VertexProjectID = &k.VertexKeyConfig.ProjectID
		} else {
			k.VertexProjectID = nil
		}
		if k.VertexKeyConfig.Region != "" {
			k.VertexRegion = &k.VertexKeyConfig.Region
		} else {
			k.VertexRegion = nil
		}
		if k.VertexKeyConfig.AuthCredentials != "" {
			k.VertexAuthCredentials = &k.VertexKeyConfig.AuthCredentials
		} else {
			k.VertexAuthCredentials = nil
		}
	} else {
		k.VertexProjectID = nil
		k.VertexRegion = nil
		k.VertexAuthCredentials = nil
	}

	if k.BedrockKeyConfig != nil {
		if k.BedrockKeyConfig.AccessKey != "" {
			k.BedrockAccessKey = &k.BedrockKeyConfig.AccessKey
		} else {
			k.BedrockAccessKey = nil
		}
		if k.BedrockKeyConfig.SecretKey != "" {
			k.BedrockSecretKey = &k.BedrockKeyConfig.SecretKey
		} else {
			k.BedrockSecretKey = nil
		}
		k.BedrockSessionToken = k.BedrockKeyConfig.SessionToken
		k.BedrockRegion = k.BedrockKeyConfig.Region
		k.BedrockARN = k.BedrockKeyConfig.ARN
		if k.BedrockKeyConfig.Deployments != nil {
			data, err := sonic.Marshal(k.BedrockKeyConfig.Deployments)
			if err != nil {
				return err
			}
			s := string(data)
			k.BedrockDeploymentsJSON = &s
		} else {
			k.BedrockDeploymentsJSON = nil
		}
	} else {
		k.BedrockAccessKey = nil
		k.BedrockSecretKey = nil
		k.BedrockSessionToken = nil
		k.BedrockRegion = nil
		k.BedrockARN = nil
		k.BedrockDeploymentsJSON = nil
	}

	if k.ApertusKeyConfig != nil {
		if k.ApertusKeyConfig.Endpoint != "" {
			k.ApertusEndpoint = &k.ApertusKeyConfig.Endpoint
		} else {
			k.ApertusEndpoint = nil
		}
	} else {
		k.ApertusEndpoint = nil
	}

	return nil
}

func (k *TableKey) AfterFind(tx *gorm.DB) error {
	if k.ModelsJSON != "" {
		if err := json.Unmarshal([]byte(k.ModelsJSON), &k.Models); err != nil {
			return err
		}
	}

	// Reconstruct Azure config if fields are present
	if k.AzureEndpoint != nil {
		azureConfig := &schemas.AzureKeyConfig{
			Endpoint:   "",
			APIVersion: k.AzureAPIVersion,
		}

		if k.AzureEndpoint != nil {
			azureConfig.Endpoint = *k.AzureEndpoint
		}

		if k.AzureDeploymentsJSON != nil {
			var deployments map[string]string
			if err := json.Unmarshal([]byte(*k.AzureDeploymentsJSON), &deployments); err != nil {
				return err
			}
			azureConfig.Deployments = deployments
		} else {
			azureConfig.Deployments = nil
		}

		k.AzureKeyConfig = azureConfig
	}

	// Reconstruct Vertex config if fields are present
	if k.VertexProjectID != nil || k.VertexRegion != nil || k.VertexAuthCredentials != nil {
		config := &schemas.VertexKeyConfig{}

		if k.VertexProjectID != nil {
			config.ProjectID = *k.VertexProjectID
		}

		if k.VertexRegion != nil {
			config.Region = *k.VertexRegion
		}
		if k.VertexAuthCredentials != nil {
			config.AuthCredentials = *k.VertexAuthCredentials
		}

		k.VertexKeyConfig = config
	}

	// Reconstruct Bedrock config if fields are present
	if k.BedrockAccessKey != nil || k.BedrockSecretKey != nil || k.BedrockSessionToken != nil || k.BedrockRegion != nil || k.BedrockARN != nil || (k.BedrockDeploymentsJSON != nil && *k.BedrockDeploymentsJSON != "") {
		bedrockConfig := &schemas.BedrockKeyConfig{}

		if k.BedrockAccessKey != nil {
			bedrockConfig.AccessKey = *k.BedrockAccessKey
		}

		bedrockConfig.SessionToken = k.BedrockSessionToken
		bedrockConfig.Region = k.BedrockRegion
		bedrockConfig.ARN = k.BedrockARN

		if k.BedrockSecretKey != nil {
			bedrockConfig.SecretKey = *k.BedrockSecretKey
		}

		if k.BedrockDeploymentsJSON != nil {
			var deployments map[string]string
			if err := json.Unmarshal([]byte(*k.BedrockDeploymentsJSON), &deployments); err != nil {
				return err
			}
			bedrockConfig.Deployments = deployments
		}

		k.BedrockKeyConfig = bedrockConfig
	}

	// Reconstruct Apertus config if fields are present
	if k.ApertusEndpoint != nil {
		apertusConfig := &schemas.ApertusKeyConfig{
			Endpoint: *k.ApertusEndpoint,
		}
		k.ApertusKeyConfig = apertusConfig
	}

	return nil
}
